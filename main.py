print("Импорт библиотек...")
from fastapi import FastAPI, File, Form, UploadFile, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tools import process_query, id2obj

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
COOKIE_NAME = "my_cookie"

history = []


@app.get("/")
async def main(request: Request, response: Response):
    global history
    history = []

    context = {"request": request}
    return templates.TemplateResponse(name="chat.html", context=context)


@app.post("/get_objects")
async def get_objects(request: Request, response: Response):
    global history

    data = await request.body()
    text = data.decode("utf-8")

    try:
        response = process_query(text, history)
        print("response:", response)
    except Exception as e:
        response = {"text": f"Ошибка: {e}"}
        print("Exception:", e)

    return response


@app.post("/get_object_info")
async def get_object_info(request: Request):
    data = await request.body()
    text = data.decode("utf-8")
    obj_id = int(text)
    obj = id2obj[obj_id]

    response = {
        "id": obj_id,
        "title": obj["title"],
        "description": obj["full_description"],
        "start_price": obj["start_price"],
        "address": obj["address"],
        "main_cat": (
            obj["categories_list"][0]
            if len(obj["categories_list"]) > 0
            else "Не указано"
        ),
        "images": obj["images"],
        "age_restriction": ["0+", "6+", "12+", "16+", "18+"][obj["age_restriction"]],
        "dates": obj["dates_list"],
    }

    return response


@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    with open(audio.filename, "wb") as file:
        content = await audio.read()
        file.write(content)
    return {"message": "Аудио файл успешно сохранен."}


if __name__ == "__main__":
    from pyngrok import ngrok
    import uvicorn

    ngrok.set_auth_token("24y35iVplRBX0a1JrFOyLn3NBQW_89A4Uwv78rMjG6hJnD7M6")

    # specify a port
    port = 8000
    ngrok_tunnel = ngrok.connect(port)

    print("Public URL:", ngrok_tunnel.public_url)
    # # finally run the app
    # uvicorn.run(app, port=port)

    print(f"Запуск...")
    uvicorn.run(app, host="127.0.0.1", port=port)
