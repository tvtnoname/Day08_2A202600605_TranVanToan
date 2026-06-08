import chainlit as cl
import requests
import json

FASTAPI_URL = "http://localhost:8000/chat"

@cl.on_chat_start
async def start():
    # 1. Khởi tạo lịch sử cuộc hội thoại
    cl.user_session.set("chat_history", [])
    
    # 2. Cài đặt bảng điều khiển RAG Settings
    settings = await cl.ChatSettings([
        cl.input_widget.Slider(
            id="top_k",
            label="Số lượng tài liệu truy xuất (top_k)",
            initial=5,
            min=1,
            max=10,
            step=1
        ),
        cl.input_widget.Slider(
            id="score_threshold",
            label="Ngưỡng điểm (score_threshold)",
            initial=0.3,
            min=0.0,
            max=1.0,
            step=0.05
        ),
        cl.input_widget.Select(
            id="rerank_method",
            label="Phương pháp Rerank",
            items={
                "cross_encoder": "Cross Encoder (Mặc định)",
                "mmr": "MMR (Độ đa dạng)",
                "rrf": "RRF (Trộn xếp hạng)",
                "none": "Không Rerank"
            },
            initial_value="cross_encoder"
        )
    ]).send()
    
    # Lưu cài đặt khởi tạo vào session
    cl.user_session.set("top_k", 5)
    cl.user_session.set("score_threshold", 0.3)
    cl.user_session.set("rerank_method", "cross_encoder")
    
    # Gửi lời chào chào mừng
    await cl.Message(
        content="👋 Xin chào! Tôi là Trợ lý Pháp luật Ma tuý & Tin tức Nghệ sĩ liên quan.\n\n"
                "Bạn có thể đặt câu hỏi về luật phòng chống ma túy hoặc tin tức tại đây."
    ).send()

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("top_k", settings["top_k"])
    cl.user_session.set("score_threshold", settings["score_threshold"])
    cl.user_session.set("rerank_method", settings["rerank_method"])

@cl.on_message
async def main(message: cl.Message):
    # 1. Lấy thông tin cài đặt hiện tại và lịch sử trò chơi từ session
    top_k = cl.user_session.get("top_k")
    score_threshold = cl.user_session.get("score_threshold")
    rerank_method = cl.user_session.get("rerank_method")
    chat_history = cl.user_session.get("chat_history")
    
    # 2. Tạo payload gửi lên FastAPI backend
    payload = {
        "query": message.content,
        "chat_history": chat_history,
        "top_k": top_k,
        "score_threshold": score_threshold,
        "rerank_method": rerank_method
    }
    
    # Hiệu ứng ba chấm động (typing indicator) HTML/CSS kích thước lớn, nhấp nháy chuyển động mượt mà
    thinking_html = """<div style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 0px;">
  <span style="width: 12px; height: 12px; background-color: currentColor; border-radius: 50%; display: inline-block; animation: thinking-bounce 1.2s infinite ease-in-out both; animation-delay: -0.32s;"></span>
  <span style="width: 12px; height: 12px; background-color: currentColor; border-radius: 50%; display: inline-block; animation: thinking-bounce 1.2s infinite ease-in-out both; animation-delay: -0.16s;"></span>
  <span style="width: 12px; height: 12px; background-color: currentColor; border-radius: 50%; display: inline-block; animation: thinking-bounce 1.2s infinite ease-in-out both;"></span>
  <style>
    @keyframes thinking-bounce {
      0%, 80%, 100% {
        transform: scale(0.4);
        opacity: 0.3;
      }
      40% {
        transform: scale(1.2);
        opacity: 1;
      }
    }
  </style>
</div>"""

    # Khởi tạo tin nhắn với hiệu ứng ba chấm động cạnh avatar/logo của chatbot
    msg = cl.Message(content=thinking_html)
    await msg.send()
    
    try:
        # Gọi API backend FastAPI
        response = requests.post(FASTAPI_URL, json=payload, timeout=30.0)
        
        if response.status_code == 200:
            res_data = response.json()
            answer = res_data.get("answer", "")
            sources = res_data.get("sources", [])
            retrieval_source = res_data.get("retrieval_source", "none")
            
            # 3. Tạo các phần tử trích dẫn (Citations)
            elements = []
            
            for i, src in enumerate(sources):
                meta = src.get("metadata", {})
                source_name = meta.get("source", f"Document {i+1}")
                
                elements.append(
                    cl.Text(
                        name=source_name, # Đặt trùng tên file để Chainlit tự động liên kết trích dẫn dạng [source_name]
                        content=f"📄 **Nguồn tham khảo:** {source_name}\n\n"
                                f"**Nội dung trích đoạn:**\n{src.get('content', '')}",
                        display="side"
                    )
                )
            
            # Cập nhật tin nhắn với câu trả lời và tài liệu tham khảo
            msg.content = answer
            msg.elements = elements
            await msg.update()
            
            # 4. Cập nhật lịch sử trò chuyện
            chat_history.append({"role": "user", "content": message.content})
            chat_history.append({"role": "assistant", "content": answer})
            cl.user_session.set("chat_history", chat_history)
            
        else:
            msg.content = f"❌ Lỗi từ API Backend (Mã lỗi {response.status_code}): {response.text}"
            await msg.update()
            
    except requests.exceptions.RequestException as e:
        msg.content = f"❌ Không thể kết nối tới API Backend ({FASTAPI_URL}). Vui lòng đảm bảo FastAPI backend đang chạy.\nChi tiết lỗi: {e}"
        await msg.update()
