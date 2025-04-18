from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 모델 및 토크나이저 로드 (CPU 최적화)
try:
    model_id = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # CPU에서 float32 권장
        device_map="cpu",
        low_cpu_mem_usage=True
    )
except Exception as e:
    raise RuntimeError(f"모델 로딩 실패: {str(e)}")

# 대화 기록 초기화
conversation_history = [
    {"role": "system", "content": "당신은 영화 추천 전문 챗봇입니다."}
]


@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "history": conversation_history}
    )


@app.post("/chat", response_class=HTMLResponse)
async def generate_response(request: Request, user_input: str = Form(...)):
    global conversation_history

    try:
        # 사용자 입력 추가
        conversation_history.append({"role": "user", "content": user_input})

        # Qwen2 채팅 템플릿 적용
        chat_text = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )

        # 모델 입력 생성
        inputs = tokenizer([chat_text], return_tensors="pt")

        # 응답 생성 (CPU 최적화 설정)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=128,  # CPU에서 128 토큰 권장
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # 응답 디코딩
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )

        # 대화 기록 업데이트
        conversation_history.append({"role": "assistant", "content": response})

        return templates.TemplateResponse(
            "chat.html",
            {"request": request, "history": conversation_history}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
