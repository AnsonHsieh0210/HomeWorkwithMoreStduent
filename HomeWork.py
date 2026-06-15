import streamlit as st
import pandas as pd
import numpy as np
import cv2
import io
import os
import gc
import pdfplumber
from PIL import Image
from pdf2image import convert_from_bytes

# 引入最新的 Gemini GenAI SDK
from google.genai import Client as GeminiClient
from google.genai import types
from pydantic import BaseModel, Field

# --- Define Pydantic Schema for Gemini ---
class GradingResult(BaseModel):
    score: float = Field(description="根據標準答案與學生作答圖片給予的得分，不可以超過該題的最高配分")
    reason: str = Field(description="詳細的評分理由與針對學生作答圖片的講評，若學生未作答或空白請說明")

# --- 核心影像處理與區域提取邏輯 ---
def order_points_robust(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def has_visible_content_in_crop(img_np, y_start, y_end, threshold_ratio=0.005):
    h, w, _ = img_np.shape
    y_start = max(0, int(y_start))
    y_end = min(h, int(y_end))
    if (y_end - y_start) < 10:
        return False
    crop_roi = img_np[y_start:y_end, 0:w]
    gray = cv2.cvtColor(crop_roi, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    non_zero_count = cv2.countNonZero(binary)
    return (non_zero_count / binary.size) > threshold_ratio

# 💡 優化點：利用 Streamlit 快取避免 5 人以上重複解析導致記憶體爆炸
@st.cache_data(show_spinner=False)
def cached_detect_extract(pdf_bytes, return_images=False):
    images = convert_from_bytes(pdf_bytes, dpi=300) # 稍微降到 300 DPI 減少記憶體消耗但保持清晰
    all_results = []
    page_blocks = {}
    page_cv_images = {}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p_idx, img in enumerate(images):
            img_np = np.array(img.convert("RGB"))
            page_cv_images[p_idx] = img_np.copy()
            h_img, w_img, _ = img_np.shape
            
            page_plumber = pdf.pages[p_idx]
            scale_x = page_plumber.width / w_img
            scale_y = page_plumber.height / h_img

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            gray[0:140, :] = 255
            gray[h_img - 180:h_img, :] = 255

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 18)

            detected_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (200, 1)), iterations=2)
            detected_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)), iterations=2)
            contours, _ = cv2.findContours(cv2.add(detected_h, detected_v), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                if w < h: w, h = h, w; angle += 90
                if abs(angle) < 5 or abs(angle - 180) < 5: angle = 0
                if w >= 400 and h >= 100:
                    box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(int)
                    candidates.append(((cx, cy), (w, h), angle, box))

            candidates = sorted(candidates, key=lambda x: x[0][1])
            page_blocks[p_idx] = []

            for (cx, cy), (w, h), angle, box_points in candidates:
                x_min, y_min = np.min(box_points, axis=0)
                x_max, y_max = np.max(box_points, axis=0)
                pdf_bbox = (x_min * scale_x + 3, y_min * scale_y + 3, x_max * scale_x - 3, y_max * scale_y - 3)
                
                extracted_text = ""
                try:
                    extracted_text = (page_plumber.within_bbox(pdf_bbox).extract_text() or "").strip()
                except Exception: pass

                rect_pts = order_points_robust(box_points)
                M = cv2.getPerspectiveTransform(rect_pts, np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32"))
                warped = cv2.warpPerspective(img_np, M, (int(w), int(h)))
                if w > 20 and h > 20: warped = warped[10:int(h)-10, 10:int(w)-10]

                page_blocks[p_idx].append({
                    "text": extracted_text, "image": Image.fromarray(warped),
                    "bbox": pdf_bbox, "img_box": (x_min, y_min, x_max, y_max)
                })

        for p_idx in range(len(images)):
            if p_idx > 0 and page_blocks[p_idx - 1] and page_blocks[p_idx]:
                last_p = page_blocks[p_idx - 1][-1]
                first_c = page_blocks[p_idx][0]
                has_img = has_visible_content_in_crop(page_cv_images[p_idx - 1], last_p["img_box"][3], page_cv_images[p_idx - 1].shape[0] - 180) or \
                          has_visible_content_in_crop(page_cv_images[p_idx], 140, first_c["img_box"][1])
                
                if not has_img:
                    last_p["text"] += "\n" + first_c["text"]
                    dst = Image.new('RGB', (max(last_p["image"].width, first_c["image"].width), last_p["image"].height + first_c["image"].height))
                    dst.paste(last_p["image"], (0, 0))
                    dst.paste(first_c["image"], (0, last_p["image"].height))
                    last_p["image"] = dst
                    first_c["is_merged_child"] = True

        for p_idx in range(len(images)):
            for b in page_blocks[p_idx]:
                if not b.get("is_merged_child", False):
                    all_results.append(b["image"] if return_images else b["text"])
    return all_results

# --- UI ---
def main():
    st.set_page_config(page_title="AI 多模態作業批改系統", layout="wide")
    st.title("📑 全自動考卷批改工作台")

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["學生姓名", "題目", "問題內容", "學生作答(影像物件)", "標準答案", "配分", "得分", "AI 評分理由"])

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("❌ 環境變數中找不到 GOOGLE_API_KEY")
        st.stop()

    try:
        gemini_client = GeminiClient()
    except Exception as e:
        st.error(f"💥 初始化 Gemini API 失敗：{e}"); return

    with st.sidebar:
        st.header("📥 上傳 PDF 來源")
        pdf_q = st.file_uploader("1. 問題內容 PDF", type="pdf")
        pdf_a = st.file_uploader("2. 標準答案 PDF", type="pdf")
        pdf_p = st.file_uploader("3. 配分 PDF", type="pdf")
        st.write("---")
        pdf_s_list = st.file_uploader("4. 學生作答 PDF (可多選)", type="pdf", accept_multiple_files=True)
        
        if st.button("🚀 開始全自動解析") and pdf_q and pdf_s_list and pdf_a and pdf_p:
            with st.spinner("正在解析 PDF 考卷結構..."):
                q_texts = cached_detect_extract(pdf_q.read(), return_images=False)
                a_texts = cached_detect_extract(pdf_a.read(), return_images=False)
                p_texts = cached_detect_extract(pdf_p.read(), return_images=False)
                
                all_student_data = []
                for pdf_s in pdf_s_list:
                    student_name = pdf_s.name.replace(".pdf", "")
                    s_images = cached_detect_extract(pdf_s.read(), return_images=True)
                    num_questions = max(len(q_texts), len(s_images), len(a_texts), len(p_texts))
                    
                    for i in range(num_questions):
                        all_student_data.append({
                            "學生姓名": student_name, "題目": f"第 {i+1} 題",
                            "問題內容": q_texts[i] if i < len(q_texts) else "",
                            "學生作答(影像物件)": s_images[i] if i < len(s_images) else None,
                            "標準答案": a_texts[i] if i < len(a_texts) else "",
                            "配分": p_texts[i] if i < len(p_texts) else "10",
                            "得分": 0.0, "AI 評分理由": ""
                        })
                st.session_state.df = pd.DataFrame(all_student_data)
                st.success(f"成功載入 {len(pdf_s_list)} 位學生的考卷。")

    # --- 主要展示面板（無排行、支援多生） ---
    if len(st.session_state.df) > 0:
        student_list = st.session_state.df["學生姓名"].unique().tolist()
        
        # 頂部直接切換學生
        selected_student = st.selectbox("👤 請選擇要檢視/批改的學生：", student_list)
        
        student_mask = st.session_state.df["學生姓名"] == selected_student
        current_student_df = st.session_state.df[student_mask].copy()

        # 功能控制按鈕
        btn_col1, btn_col2, score_col = st.columns([1.5, 1.5, 2])
        with btn_col1:
            run_all_ai = st.button("🚀 一鍵批改全班所有同學", width="stretch")
        with btn_col2:
            run_ai = st.button(f"🤖 僅批改當前學生 ({selected_student})", width="stretch")
        with score_col:
            try:
                current_score = pd.to_numeric(current_student_df["得分"]).sum()
            except Exception: current_score = 0
            st.markdown(f"### 🎯 {selected_student} 總分：{current_score:.1f}")

        st.write("---")

        col1, col2 = st.columns([7, 5])
        with col1:
            st.subheader("📝 得分明細與修正")
            display_df = current_student_df.copy()
            display_df["學生作答(影像物件)"] = display_df["學生作答(影像物件)"].apply(
                lambda x: "📷 影像已就緒" if x is not None else "⚠️ 無影像"
            )

            edited_display_df = st.data_editor(
                display_df.drop(columns=["學生姓名"]),
                num_rows="dynamic", width="stretch", height=350,
                key=f"editor_{selected_student}"
            )
            
            if len(edited_display_df) == len(current_student_df):
                indices = current_student_df.index
                st.session_state.df.loc[indices, "得分"] = edited_display_df["得分"].values
                st.session_state.df.loc[indices, "AI 評分理由"] = edited_display_df["AI 評分理由"].values
                st.session_state.df.loc[indices, "問題內容"] = edited_display_df["問題內容"].values
                st.session_state.df.loc[indices, "標準答案"] = edited_display_df["標準答案"].values
                st.session_state.df.loc[indices, "配分"] = edited_display_df["配分"].values

            st.write("---")
            q_list = current_student_df["題目"].tolist()
            selected_q_name = st.selectbox("🔍 請選擇你想在右側複核的題號：", q_list, index=0)
            selected_idx = q_list.index(selected_q_name) if selected_q_name in q_list else 0

        with col2:
            st.subheader("🔍 盲區視覺複核面板")
            if selected_idx is not None and selected_idx < len(current_student_df):
                row_data = current_student_df.iloc[selected_idx]
                st.markdown(f"#### 📋 當前檢視：**{selected_student} - {row_data['題目']}**")
                st.markdown(f"**問題：** {row_data['問題內容']}")
                st.markdown(f"**標準答案：** {row_data['標準答案']}")
                
                img_obj = row_data["學生作答(影像物件)"]
                if img_obj is not None:
                    st.image(img_obj, width="stretch", caption=f"{selected_student} {row_data['題目']} 裁切影像")
                else:
                    st.warning("⚠️ 該題無影像")

        # --- 批改全班邏輯 (優化記憶體放寬限制) ---
        if run_all_ai:
            temp_df = st.session_state.df.copy()
            status_text = st.empty()
            total_rows = len(temp_df)
            
            for count, (index, row) in enumerate(temp_df.iterrows()):
                status_text.markdown(f"🚀 **批改進度：正在批改 [{row['學生姓名']}] 的 {row['題目']} ({count+1}/{total_rows})...**")
                student_img = row["學生作答(影像物件)"]
                
                if student_img is not None:
                    prompt = f"你是一位溫和的審查老師。評分原則為「從寬給分」。\n\n【單題資訊】\n問題內容：{row['問題內容']}\n標準答案：{row['標準答案']}\n最高配分：{row['配分']} 分。\n"
                    try:
                        response = gemini_client.models.generate_content(
                            model='gemini-2.5-pro', contents=[prompt, student_img],
                            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=GradingResult, temperature=0.1),
                        )
                        result: GradingResult = response.parsed
                        temp_df.at[index, "得分"] = float(result.score)
                        temp_df.at[index, "AI 評分理由"] = result.reason
                    except Exception: pass
                else:
                    temp_df.at[index, "得分"] = 0.0
                    temp_df.at[index, "AI 評分理由"] = "無影像物件"
                
                # 💡 每改完一題就釋放不用的圖片記憶體，防止 5 人以上當機
                if count % 5 == 0:
                    gc.collect()

            status_text.empty()
            st.session_state.df = temp_df
            st.success("🎉 全班考卷批改完畢！")
            st.rerun()

        # --- 批改單生邏輯 ---
        if run_ai:
            temp_df = st.session_state.df.copy()
            target_indices = temp_df[temp_df["學生姓名"] == selected_student].index
            status_text = st.empty()
            
            for count, index in enumerate(target_indices):
                status_text.markdown(f"⏳ **正在批改 {selected_student} 第 {count + 1} 題...**")
                row = temp_df.loc[index]
                student_img = row["學生作答(影像物件)"]
                
                if student_img is not None:
                    prompt = f"你是一位溫和的審查老師。評分原則為「從寬給分」。\n\n【單題資訊】\n問題內容：{row['問題內容']}\n標準答案：{row['標準答案']}\n最高配分：{row['配分']} 分。\n"
                    try:
                        response = gemini_client.models.generate_content(
                            model='gemini-2.5-pro', contents=[prompt, student_img],
                            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=GradingResult, temperature=0.1),
                        )
                        result: GradingResult = response.parsed
                        temp_df.at[index, "得分"] = float(result.score)
                        temp_df.at[index, "AI 評分理由"] = result.reason
                    except Exception as e:
                        st.warning(f"第 {count+1} 題評分錯誤: {e}")
                else:
                    temp_df.at[index, "得分"] = 0.0
                    temp_df.at[index, "AI 評分理由"] = "未偵測到影像。"
            
            status_text.empty()
            st.session_state.df = temp_df
            st.success(f"🎉 {selected_student} 批改完成！")
            st.rerun()
    else:
        st.info("💡 請在左側欄上傳資料並點擊開始全自動解析。")

if __name__ == "__main__":
    main()
