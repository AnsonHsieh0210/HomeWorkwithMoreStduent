import streamlit as st
import pandas as pd
import numpy as np
import cv2
import io
import os
import pdfplumber
from PIL import Image
from pdf2image import convert_from_bytes

# 引入最新的 Gemini GenAI SDK (負責視覺辨識與結構化批改)
from google.genai import Client as GeminiClient
from google.genai import types
from pydantic import BaseModel, Field

# --- Define Pydantic Schema for Gemini Structured Output ---

class GradingResult(BaseModel):
    score: float = Field(description="根據標準答案與學生作答圖片給予的得分，不可以超過該題的最高配分")
    reason: str = Field(description="詳細的評分理由與針對學生作答圖片的講評，若學生未作答或空白請說明")

# --- 核心影像處理與區域提取邏輯 ---

def order_points_robust(pts):
    """強健的點排序：左上, 右上, 右下, 左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # tl
    rect[2] = pts[np.argmax(s)]    # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect

def has_visible_content_in_crop(img_np, y_start, y_end, threshold_ratio=0.005):
    """針對掃描檔的盲區內容偵測"""
    h, w, _ = img_np.shape
    y_start = max(0, int(y_start))
    y_end = min(h, int(y_end))
    
    if (y_end - y_start) < 10:
        return False
        
    crop_roi = img_np[y_start:y_end, 0:w]
    gray = cv2.cvtColor(crop_roi, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    non_zero_count = cv2.countNonZero(binary)
    total_pixels = binary.size
    black_pixel_ratio = non_zero_count / total_pixels
    
    return black_pixel_ratio > threshold_ratio

def detect_and_extract_blocks(pdf_bytes, min_w=400, min_h=100, return_images=False):
    """
    結合 OpenCV 定位。
    💡 針對文字 PDF (Q, A, P) 提取純文字；針對學生作答 (S) 則裁切並回傳 PIL.Image 物件列表。
    """
    images = convert_from_bytes(pdf_bytes, dpi=350)
    all_results = [] 
    
    page_blocks = {} 
    page_cv_images = {} 

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p_idx, img in enumerate(images):
            img_np = np.array(img.convert("RGB"))
            page_cv_images[p_idx] = img_np.copy()
            h_img, w_img, _ = img_np.shape
            
            page_plumber = pdf.pages[p_idx]
            w_pdf, h_pdf = page_plumber.width, page_plumber.height
            
            scale_x = w_pdf / w_img
            scale_y = h_pdf / h_img

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            ignore_top_px = 140      # 忽略頁首高度
            ignore_bottom_px = 180   # 忽略頁尾與頁碼高度
            
            if ignore_top_px > 0:
                gray[0:ignore_top_px, :] = 255
            if ignore_bottom_px > 0:
                gray[h_img - ignore_bottom_px:h_img, :] = 255

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 21, 18)

            h_k = max(60, int(min_w * 0.5)) 
            v_k = max(30, int(min_h * 0.6))
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_k, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_k))
            
            detected_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            detected_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            detected_lines = cv2.add(detected_h, detected_v)

            contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                if w < h:
                    w, h = h, w
                    angle += 90
                
                if abs(angle) < 5 or abs(angle - 180) < 5: angle = 0
                
                if w >= min_w and h >= min_h:
                    box = cv2.boxPoints(((cx, cy), (w, h), angle))
                    box = box.astype(int)
                    candidates.append(((cx, cy), (w, h), angle, box))

            candidates = sorted(candidates, key=lambda x: x[0][1])
            page_blocks[p_idx] = []

            for (cx, cy), (w, h), angle, box_points in candidates:
                x_min, y_min = np.min(box_points, axis=0)
                x_max, y_max = np.max(box_points, axis=0)
                
                pdf_bbox = (
                    x_min * scale_x + 3, 
                    y_min * scale_y + 3, 
                    x_max * scale_x - 3, 
                    y_max * scale_y - 3
                )
                
                extracted_text = ""
                try:
                    crop = page_plumber.within_bbox(pdf_bbox)
                    extracted_text = (crop.extract_text() or "").strip()
                except Exception:
                    extracted_text = ""

                cropped_img = None
                rect_pts = order_points_robust(box_points)
                dst_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
                warped = cv2.warpPerspective(img_np, M, (int(w), int(h)))
                
                pad = 10
                if w > pad*2 and h > pad*2:
                    warped = warped[pad:int(h)-pad, pad:int(w)-pad]
                
                cropped_img = Image.fromarray(warped)

                page_blocks[p_idx].append({
                    "text": extracted_text,
                    "image": cropped_img,
                    "bbox": pdf_bbox,
                    "img_box": (x_min, y_min, x_max, y_max)
                })

        total_pages = len(images)
        for p_idx in range(total_pages):
            if p_idx > 0 and len(page_blocks[p_idx - 1]) > 0 and len(page_blocks[p_idx]) > 0:
                prev_page_plumber = pdf.pages[p_idx - 1]
                curr_page_plumber = pdf.pages[p_idx]
                
                last_block_prev_page = page_blocks[p_idx - 1][-1]
                first_block_curr_page = page_blocks[p_idx][0]
                
                text_below_last_table = ""
                text_above_first_table = ""
                try:
                    bottom_crop = prev_page_plumber.crop((0, last_block_prev_page["bbox"][3], prev_page_plumber.width, prev_page_plumber.height))
                    text_below_last_table = bottom_crop.extract_text() or ""
                    
                    top_crop = curr_page_plumber.crop((0, 0, curr_page_plumber.width, first_block_curr_page["bbox"][0]))
                    text_above_first_table = top_crop.extract_text() or ""
                except Exception:
                    pass
                
                has_digital_text_between = bool(text_below_last_table.strip() or text_above_first_table.strip())
                
                prev_img_np = page_cv_images[p_idx - 1]
                curr_img_np = page_cv_images[p_idx]
                
                _, _, _, last_y_max = last_block_prev_page["img_box"]
                _, first_y_min, _, _ = first_block_curr_page["img_box"]
                
                has_scanned_content_below = has_visible_content_in_crop(prev_img_np, last_y_max, prev_img_np.shape[0] - ignore_bottom_px)
                has_scanned_content_above = has_visible_content_in_crop(curr_img_np, ignore_top_px, first_y_min)
                
                has_image_content_between = has_scanned_content_below or has_scanned_content_above
                
                if (not has_digital_text_between) and (not has_image_content_between):
                    last_block_prev_page["text"] += "\n" + first_block_curr_page["text"]
                    
                    img1 = last_block_prev_page["image"]
                    img2 = first_block_curr_page["image"]
                    dst = Image.new('RGB', (max(img1.width, img2.width), img1.height + img2.height))
                    dst.paste(img1, (0, 0))
                    dst.paste(img2, (0, img1.height))
                    last_block_prev_page["image"] = dst
                    
                    first_block_curr_page["is_merged_child"] = True

        for p_idx in range(total_pages):
            for block in page_blocks[p_idx]:
                if not block.get("is_merged_child", False):
                    if return_images:
                        all_results.append(block["image"]) 
                    else:
                        all_results.append(block["text"])  

    return all_results

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="AI 多模態多生評分系統", layout="wide")
    st.title("📑 全自動考卷批改工作台 (多生批閱版)")

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["學生姓名", "題目", "問題內容", "學生作答(影像物件)", "標準答案", "配分", "得分", "AI 評分理由"])

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("❌ 環境變數中找不到 GOOGLE_API_KEY，請確認 Streamlit Secrets 設定。")
        st.stop()

    try:
        gemini_client = GeminiClient() 
    except Exception as e:
        st.error(f"💥 初始化 Gemini API 失敗：{e}")
        return

    with st.sidebar:
        st.header("📥 上傳 PDF 來源")
        pdf_q = st.file_uploader("1. 問題內容 PDF", type="pdf")
        pdf_a = st.file_uploader("2. 標準答案 PDF", type="pdf")
        pdf_p = st.file_uploader("3. 配分 PDF", type="pdf")
        
        st.write("---")
        pdf_s_list = st.file_uploader("4. 學生作答 PDF (可一次選取多個檔案)", type="pdf", accept_multiple_files=True)
        
        if st.button("🚀 開始全自動解析") and pdf_q and pdf_s_list and pdf_a and pdf_p:
            with st.spinner("正在執行跨學生結構化座標對齊與裁切..."):
                q_bytes = pdf_q.read()
                a_bytes = pdf_a.read()
                p_bytes = pdf_p.read()

                q_texts = detect_and_extract_blocks(q_bytes, return_images=False)
                a_texts = detect_and_extract_blocks(a_bytes, return_images=False)
                p_texts = detect_and_extract_blocks(p_bytes, return_images=False)
                
                all_student_data = []
                
                for pdf_s in pdf_s_list:
                    student_name = pdf_s.name.replace(".pdf", "")
                    s_bytes = pdf_s.read()
                    s_images = detect_and_extract_blocks(s_bytes, return_images=True)
                    
                    num_questions = max(len(q_texts), len(s_images), len(a_texts), len(p_texts))
                    
                    for i in range(num_questions):
                        all_student_data.append({
                            "學生姓名": student_name,
                            "題目": f"第 {i+1} 題",
                            "問題內容": q_texts[i] if i < len(q_texts) else "",
                            "學生作答(影像物件)": s_images[i] if i < len(s_images) else None, 
                            "標準答案": a_texts[i] if i < len(a_texts) else "",
                            "配分": p_texts[i] if i < len(p_texts) else "10",
                            "得分": 0.0,
                            "AI 評分理由": ""
                        })
                
                st.session_state.df = pd.DataFrame(all_student_data)
                st.success(f"解析完成！已成功載入 {len(pdf_s_list)} 位學生的考卷區塊。")

    # --- 主要展示區塊 ---
    if len(st.session_state.df) > 0:
        
        # 🌟 建立兩個分頁，切換不同的視角
        tab_summary, tab_detail = st.tabs(["🏆 全班成績大考查 (總分與每題明細)", "🔍 盲區視覺複核工作台"])
        
        # ==========================================
        # 分頁 1：全班成績大考查 (解決你的核心痛點)
        # ==========================================
        with tab_summary:
            st.subheader("📊 班級成績總覽與章節題得分")
            
            # 1. 建立控制與一鍵自動化按鈕
            btn_col1, btn_col2 = st.columns([1, 3])
            with btn_col1:
                run_all_ai = st.button("🚀 一鍵批改全班考卷", use_container_width=True, key="summary_run_all")
            with btn_col2:
                st.caption("💡 點擊按鈕將調用 Gemini 2.5 Pro 對全部學生、所有題目進行批改。")
            
            # 計算每位學生的總分
            # 預先處理配分與得分為數字型態以防加總出錯
            df_calc = st.session_state.df.copy()
            df_calc["得分"] = pd.to_numeric(df_calc["得分"], errors="coerce").fillna(0.0)
            df_calc["配分"] = pd.to_numeric(df_calc["配分"], errors="coerce").fillna(0.0)
            
            summary_scores = df_calc.groupby("學生姓名")["得分"].sum().reset_index()
            summary_scores = summary_scores.sort_values(by="得分", ascending=False)
            
            st.markdown("---")
            st.markdown("### 👥 學生總分排行榜")
            
            # 使用更簡潔好看的表格列出所有同學總分
            st.dataframe(
                summary_scores,
                column_config={
                    "學生姓名": st.column_config.TextColumn("學生姓名", width="medium"),
                    "得分": st.column_config.NumberColumn("目前總得分", format="%.1f 分"),
                },
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            st.markdown("### 📑 各生每題得分詳細清單")
            st.info("💡 點擊下方同學的名字區塊（Expander），即可直接展開查看該生「每一題的分數」與「評分理由」！")
            
            # 遍歷每位學生，建立專屬的可折疊區塊
            for student in summary_scores["學生姓名"].tolist():
                student_mask = df_calc["學生姓名"] == student
                student_df = df_calc[student_mask]
                student_total = summary_scores[summary_scores["學生姓名"] == student]["得分"].values[0]
                
                with st.expander(f"👤 {student} ─── 總得分：{student_total:.1f} 分", expanded=False):
                    
                    # 建立精簡版的每題得分 DataFrame
                    brief_detail = student_df[["題目", "配分", "得分", "AI 評分理由"]].copy()
                    
                    # 透過 data_editor 展現，不僅能看，老師還能直接在格子內手動改分數！
                    edited_brief = st.data_editor(
                        brief_detail,
                        use_container_width=True,
                        hide_index=True,
                        key=f"summary_edit_{student}",
                        column_config={
                            "題目": st.column_config.TextColumn("題號", disabled=True),
                            "配分": st.column_config.NumberColumn("最高配分", disabled=True, format="%d 分"),
                            "得分": st.column_config.NumberColumn("得分", min_value=0.0, max_value=100.0, format="%.1f"),
                            "AI 評分理由": st.column_config.TextColumn("AI 評分理由/講評", width="large")
                        }
                    )
                    
                    # 如果老師在總覽區手動修改了分數，即時回填數據源
                    if not edited_brief.equals(brief_detail):
                        st.session_state.df.loc[student_df.index, "得分"] = edited_brief["得分"].values
                        st.session_state.df.loc[student_df.index, "AI 評分理由"] = edited_brief["AI 評分理由"].values
                        st.rerun()

        # ==========================================
        # 分頁 2：原有的盲區視覺複核工作台 (看單題題目、對齊標準答案與影像)
        # ==========================================
        with tab_detail:
            student_list = st.session_state.df["學生姓名"].unique().tolist()
            selected_student = st.selectbox("👤 請選擇要進行視覺核對的學生：", student_list, key="detail_student_select")
            
            student_mask = st.session_state.df["學生姓名"] == selected_student
            current_student_df = st.session_state.df[student_mask].copy()

            col1, col2 = st.columns([7, 5])
            
            with col1:
                st.subheader(f"📝 {selected_student} 的單題細節")
                
                display_df = current_student_df.copy()
                display_df["學生作答(影像物件)"] = display_df["學生作答(影像物件)"].apply(
                    lambda x: "📷 影像已就緒" if x is not None else "⚠️ 無影像"
                )

                edited_display_df = st.data_editor(
                    display_df.drop(columns=["學生姓名"]),
                    num_rows="dynamic",
                    use_container_width=True,
                    height=300,
                    key=f"editor_{selected_student}"
                )
                
                if len(edited_display_df) == len(current_student_df):
                    indices = current_student_df.index
                    st.session_state.df.loc[indices, "得分"] = edited_display_df["得分"].values
                    st.session_state.df.loc[indices, "AI 評分理由"] = edited_display_df["AI 評分理由"].values
                    st.session_state.df.loc[indices, "問題內容"] = edited_display_df["問題內容"].values
                    st.session_state.df.loc[indices, "標準答案"] = edited_display_df["標準答案"].values
                    st.session_state.df.loc[indices, "配分"] = edited_display_df["配分"].values

                current_score = pd.to_numeric(st.session_state.df[student_mask]["得分"]).sum()
                
                btn_col, score_col = st.columns([1, 1])
                with btn_col:
                    run_ai = st.button(f"🤖 批改這位學生 ({selected_student})", use_container_width=True, key="run_single_ai")
                with score_col:
                    st.markdown(f"### 🎯 該生總分：{current_score:.1f}")

                st.write("---")
                st.subheader("🔍 題號影像定位切換")
                q_list = current_student_df["題目"].tolist()
                selected_q_name = st.selectbox("請選擇你想在右側複核的題目：", q_list, index=0, key="q_select_detail")
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
                        st.image(img_obj, use_container_width=True, caption=f"{selected_student} {row_data['題目']} 盲區裁剪影像")
                    else:
                        st.warning("⚠️ 該題無對應的學生作答影像")

        # ==========================================
        # 共享的一鍵批改所有人核心邏輯 (兩個分頁按鈕皆能觸發)
        # ==========================================
        if run_all_ai:
            temp_df = st.session_state.df.copy()
            status_text = st.empty()
            total_rows = len(temp_df)
            
            for count, (index, row) in enumerate(temp_df.iterrows()):
                status_text.markdown(f"🚀 **一鍵總批改中：正在批改 [{row['學生姓名']}] 的 {row['題目']} ({count+1}/{total_rows})...**")
                student_img = row["學生作答(影像物件)"]
                
                if student_img is not None:
                    prompt = (
                        f"你是一位溫和、具鼓勵性質的專業審查老師。目前正在批改學生的作答內容，評分核心原則為「從寬給分」。\n\n"
                        f"【單題題目資訊】\n問題內容：{row['問題內容']}\n標準答案：{row['標準答案']}\n最高配分：{row['配分']} 分。\n\n"
                    )
                    try:
                        response = gemini_client.models.generate_content(
                            model='gemini-2.5-pro',  
                            contents=[prompt, student_img],
                            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=GradingResult, temperature=0.1),
                        )
                        result: GradingResult = response.parsed
                        temp_df.at[index, "得分"] = float(result.score)
                        temp_df.at[index, "AI 評分理由"] = result.reason
                    except Exception:
                        pass
                else:
                    temp_df.at[index, "得分"] = 0.0
                    temp_df.at[index, "AI 評分理由"] = "無影像物件"
            
            status_text.empty()
            st.session_state.df = temp_df
            st.success("🎉 所有學生全部批改完畢！")
            st.rerun()

        # 共享的單人批改核心邏輯
        if 'run_ai' in locals() and run_ai:
            temp_df = st.session_state.df.copy()
            target_indices = temp_df[temp_df["學生姓名"] == selected_student].index
            status_text = st.empty()
            
            for count, index in enumerate(target_indices):
                status_text.markdown(f"⏳ **Gemini 正在批改 {selected_student} 第 {count + 1} 題...**")
                row = temp_df.loc[index]
                student_img = row["學生作答(影像物件)"]
                
                if student_img is not None:
                    prompt = (
                        f"你是一位溫和、具鼓勵性質的專業審查老師。目前正在批改學生的作答內容，評分核心原則為「從寬給分」。\n\n"
                        f"【單題題目資訊】\n問題內容：{row['問題內容']}\n標準答案：{row['標準答案']}\n最高配分：{row['配分']} 分。\n\n"
                        f"【任務說明】\n1. 審視圖片內學生的手寫答案。\n2. 給予 0 到 {row['配分']} 之間的合理分數。\n3. 詳細列出評分理由。\n"
                    )
                    try:
                        response = gemini_client.models.generate_content(
                            model='gemini-2.5-pro',  
                            contents=[prompt, student_img],
                            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=GradingResult, temperature=0.1),
                        )
                        result: GradingResult = response.parsed
                        temp_df.at[index, "得分"] = float(result.score)
                        temp_df.at[index, "AI 評分理由"] = result.reason
                    except Exception as e:
                        st.warning(f"第 {count+1} 題評分錯誤: {e}")
                else:
                    temp_df.at[index, "得分"] = 0.0
                    temp_df.at[index, "AI 評分理由"] = "未偵測到學生作答圖片，以 0 分計算。"
            
            status_text.empty()
            st.session_state.df = temp_df
            st.success(f"🎉 {selected_student} 批改完成！")
            st.rerun()

    else:
        st.info("💡 請在左側欄上傳題目、標準答案、配分以及「多位學生」的作答 PDF，並點擊開始全自動解析。")

if __name__ == "__main__":
    main()
