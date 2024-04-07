import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models.gigachat import GigaChat
import json
import re
import time

with open("id2obj.pickle", "rb") as file:
    id2obj = pickle.load(file)

db_embeddings = np.load("db_embeddings.npy")

with open(
    "index2obj_id.pickle",
    "rb",
) as file:
    index2obj_id = pickle.load(file)

cat_counter = {}

for event in id2obj.values():
    for cat in event["categories"]:
        cat_counter[cat] = cat_counter.get(cat, 0) + 1

cat_counter_list = list(cat_counter.keys())
cat_counter_list.sort()


device = "cuda"

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
).to(device)


auth = "MDRhODc1YjMtZDMxNi00OTJmLWE1M2MtNjYxNDJlNjUwMWRhOmJhOGUyZDljLTkwMjktNDQ3Zi05ZjAyLWM0ZDNhZDhlOWViOA=="

giga = GigaChat(credentials=auth, model="GigaChat:latest", verify_ssl_certs=False)


def get_new_history(gen_type="parse", return_type="object"):
    if gen_type == "parse":
        sys_message = f"""Ты являешься профессиональным туристическим помощником-ассистентом. Твои обязанности - выполнять задачи по структуризации текста (получения определённой информации из текста). 
        Каждый твой ответ должен быть исключительно в формате JSON с полями: is_search_query, answer, categories, tags, age_restriction, dates, cost_estimate, cleaned_query.
        Вопросы будут требовать указывать категории для заданных текстов. Для этого выбирай категории из следующего списка: {', '.join(cat_counter_list)}."""
    else:
        sys_message = f"""Ты являешься профессиональным туристическим помощником-ассистентом. Твоя задача - для каждого объекта (место или мероприятие) из заданного списка дать привлекательное длинное обоснование , почему пользователю нужно пойти именно туда, учитывая запрос пользователя.
                Дай ответ исключительно в формате JSON: список из обоснований (строк)."""

    if return_type == "object":
        msgs = [SystemMessage(content=sys_message)]
    else:
        msgs = [{"role": "system", "content": sys_message}]

    return msgs


generation_time = 0


def get_answer(history):
    global generation_time

    t1 = time.time()
    answer = giga(history)
    t2 = time.time()
    generation_time = generation_time + t2 - t1

    return answer.content


def get_query_parse_prompt(text, use_add_text=False):
    if use_add_text:
        add_text = "Учитывая предыдущие вопросы и ответы, тебе необходимо для данного запроса определить:"
    else:
        add_text = "Тебе необходимо для данного запроса определить:"
    #
    prompt = f"""Тебе будет дан текстовой поисковой запрос пользователя. {add_text}
    1. 6 категорий на русской языке (поле "categories" - список из строк).
    2. как можно больше тегов на русском языке (поле "tags" - список из строк).
    3. возрастное ограничение, если оно указано. Если возрастное ограничение не указано напрямую, то пытайся вычислить его, ориентируясь на запрос пользователя (поле "age_restriction" - строка с одним из значений: "0+", "6+", "12+", "16+", "18+"; null - если возраст не указан).
    4. даты, если они указаны. Если даты напрямую не указаны (например, "завтра", "на этих выходных", "в пятницу" и так далее), то твоя задача вычислить эти даты, ориентируясь на текущую дату. Например, для текущей даты 7.04.24 (воскресенье) и запроса "я хочу погулять на следующих выходных" правильный ответом для поля "dates" будет ["13.04.2024", "14.04.2024"] (поле "dates" - список из строк, представляющий собой предполагаемые даты в формате "%d.%m.%Y"; null - если даты не указаны).
    5. примерная ценовая категория (поле "cost_estimate" - null, если про цену не указано в запросе или строка с одним из значений: "бесплатно" - 0 рублей, "дёшево" - до 500 рублей, "средне" - от 500 рублей до 2000 рублей, "дорого" - от 2000 рублей).
    6. очищенный (от времени, дат и цен) запрос пользователя, прямое предпочтение пользователя (поле "cleaned_query" - строка из 2-3 слов).

    Текущий город: Москва.
    Текущая дата (сегодня): 07.04.2024 (7 апреля 2024 года).
    День недели: воскресенье.
    Запрос пользователя: "{text}".
    Дай ответ исключительно в формате JSON без лишних объяснений, чтобы его можно было считать программным путём."""

    # 1. Является ли запрос поисковым или уточняющим. Цель поискового запроса - поиск мест и мероприятий. Цель уточняющего запроса - спросить или уточнить информацию предыдущих ответов чата, либо просто поощаться (поле "is_search_query" - булево значение).
    # 2. Ответ на уточняющий запрос, если данный запрос является уточняющим (поле "answer" - строка, если запрос уточняющий или null, если запрос является поисковым).

    return prompt


def parse_json(text, braces_type="{}"):  # !!! Доделать
    text = re.sub(r"\"+", '"', text)
    start_index = text.find(braces_type[0])
    end_index = text.rfind(braces_type[1]) + 1
    json_substring = text[start_index:end_index]

    dv_counter = sum([1 for c in json_substring if c == ":"])
    za_counter = sum([1 for c in json_substring if c == ","])

    if braces_type == "[]" and za_counter > dv_counter:
        json_substring = json_substring.replace(":", ",")

    json_data = json.loads(json_substring)
    return json_data


def get_sims(text):
    embeddings = model.encode([text])
    sims = embeddings @ db_embeddings.T
    sims = sims[0]

    obj_id2sim = {i: -100000 for i in range(len(id2obj))}

    for index, sim in enumerate(sims):
        obj_id = index2obj_id[index]

        obj_id2sim[obj_id] = max(obj_id2sim[obj_id], sim)

    return obj_id2sim


# получить возможные текстовы запросы для каждого объекта
from tqdm.auto import tqdm

# "бесплатно" - 0 рублей, "дёшево" - до 500 рублей, "средне" - от 500 рублей до 2000 рублей, "дорого" - от 2000 рублей).
cost_estimate2num = {
    "бесплатно": 0,
    "дёшево": 1,
    "средне": 2,
    "дорого": 3,
}

age_restriction2num = {
    "0+": 0,
    "6+": 1,
    "12+": 2,
    "16+": 3,
    "18+": 4,
}


def search_objects(search_query, n_limit=5):
    """
    search_query:
        text: str,
        categories: list[str],
        tags: list[str],
        age_restriction: int,
        near_to_address: str,
        cost_estimate: str,
        dates: list[str],
        times: list[str]
    """
    categories = set(search_query["categories"])
    tags = set(search_query["categories"])
    dates = set() if (search_query["dates"] is None) else set(search_query["dates"])

    age_restriction = search_query["age_restriction"]
    if age_restriction:
        age_restriction = age_restriction2num[age_restriction]

    cost_estimate = search_query["cost_estimate"]
    if cost_estimate:
        cost_estimate_num = cost_estimate2num[cost_estimate]
    else:
        cost_estimate_num = None

    obj_id2sim = get_sims(search_query["cleaned_query"])

    scores = []
    for obj_id in range(len(id2obj)):
        obj = id2obj[obj_id]
        sim = obj_id2sim[obj_id]

        if age_restriction:
            is_valid_age_restriction = age_restriction >= obj["age_restriction"]
            if not is_valid_age_restriction:
                continue

        # dates
        if len(dates) > 0:
            inter_dates = dates & obj["dates"]
            if len(inter_dates) == 0:
                continue

        # cost_estimate
        if cost_estimate_num:
            cost_estimate_score = -abs(cost_estimate_num - obj["cost_estimate_num"])
        else:
            cost_estimate_score = 0

        inter_cats = categories & obj["categories"]
        n_inter_cats = len(inter_cats)

        inter_tags = tags & obj["tags"]
        n_inter_tags = len(inter_tags)

        score = (
            sim + n_inter_cats / 2 + 2 * n_inter_tags + cost_estimate_score
        )  # + estimate_time_score
        if score < 0:
            continue
        scores.append(
            (
                obj_id,
                score,
                sim,
                n_inter_cats,
                n_inter_tags,
                cost_estimate_score,
            )
        )

    scores.sort(key=lambda el: -el[1])
    print("TOP score:", scores[0])
    obj_ids = [el[0] for el in scores[:n_limit]]

    objs = [id2obj[obj_id] for obj_id in obj_ids]
    print("TOP object:", objs[0]["title"], objs[0]["tags"], objs[0]["categories"])

    return objs


def get_object_justification_messages(objects, query):
    messages = get_new_history(gen_type="justification")

    objects_description = [
        f"""Описание объекта с номером {i + 1} (\"\"\"{event['title']}\"\"\"): \"\"\"{event['description']}\"\"\";"""
        for i, event in enumerate(objects)
    ]

    objects_description = "\n".join(objects_description)

    prompt = f"""Тебе даны описания мест и мероприятий, полученных на основе следующего текстового запроса пользователя: "{query}".
    Описания мест и мероприятий:
    {objects_description}

    Твоя задача - для каждого объекта (место или мероприятие) дать привлекательное обоснование, почему пользователю нужно пойти именно туда, основываясь на запросе пользователя.
    Дай ответ исключительно в формате JSON: массив из обоснований (строк).
    """

    messages.append(HumanMessage(content=prompt))

    return messages


def process_query(query, history=[]):
    query = query.strip().lower()
    if len(history) == 0:
        history.append(get_new_history()[0])

    prompt = get_query_parse_prompt(query, use_add_text=(len(history) > 1))

    history.append(HumanMessage(content=prompt))

    answer1 = get_answer(history)

    if "Не люблю менять тему разговора, но вот сейчас тот самый случай." in answer1:
        response = {
            "text": "Не люблю менять тему разговора, но вот сейчас тот самый случай.",
        }
        return response

    print("#######################")
    print("answer1", answer1)
    print("#######################")
    search_query = parse_json(answer1)
    if not ("cleaned_query" in search_query):
        search_query["cleaned_query"] = query.strip().lower()

    # history.append(AIMessage(content=answer))

    # if not search_query["is_search_query"]:
    #     response = {
    #         "text": search_query["answer"],
    #     }
    #     ai_content = str(response)
    #     history.append(AIMessage(content=ai_content))  #!!!!!!!!!!
    #     return response, history

    objs = search_objects(search_query)
    # messages = get_object_justification_messages(objs, query)

    # answer2 = get_answer(messages)
    # if "Не люблю менять тему разговора, но вот сейчас тот самый случай." in answer2:
    #     response = {
    #         "text": "Не люблю менять тему разговора, но вот сейчас тот самый случай.",
    #     }
    #     return response
    # print("#######################")
    # print("answer2", answer2)
    print("#######################")
    # justifications = parse_json(answer2, braces_type="[]")

    json_objs = []
    # for obj, justification in zip(objs, justifications):
    for obj in objs:
        json_objs.append(
            {
                "id": obj["id"],
                "title": obj["title"],
                "description": obj["description"],
                "image_src": (
                    obj["images"][0]
                    if len(obj["images"]) > 0
                    else "https://i.imgur.com/fJCx4xp.png"
                ),
            }
        )

    response = {
        "text": "У меня есть несколько вариантов для вас:",
        "objects": json_objs,
    }

    history.append(AIMessage(content=answer1))

    #!!!!!!!!!!

    return response


if __name__ == "__main__":
    response, history = process_query("я хочу пойти потанцевать")

    print(generation_time)
    print(history)
    print(response)

    response, history = process_query("я хочу побегать", history)
    print(history)
    print(response)
