import re

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama


# ============================================================
# 1. DECOMPOSER LA QUESTION
# ============================================================

decompose_prompt = PromptTemplate.from_template(
    """Décompose la question suivante en exactement 3 sous-questions.

Retourne uniquement une liste numérotée :

1. ...
2. ...
3. ...

Question : {question}
"""
)

model_decompose = ChatOllama(
    model="llama3.2",
    temperature=0
)

decomposer = decompose_prompt | model_decompose


# ============================================================
# 2. RECUPERER LES SOUS-QUESTIONS
# ============================================================

def parse_numbered_subquestions(message):

    text = getattr(message, "content", str(message)).strip()

    lines = re.split(r"\r?\n", text)

    subquestions = []

    for line in lines:

        match = re.match(
            r"\s*\d+\s*[.)]\s*(.*\S.*)$",
            line
        )

        if match:
            subquestions.append(
                match.group(1).strip()
            )

    if not subquestions and text:
        subquestions = [text]

    return subquestions[:3]


parse_subquestions = RunnableLambda(
    parse_numbered_subquestions
)


# ============================================================
# 3. REPONDRE AUX SOUS-QUESTIONS
# ============================================================

answer_prompt = PromptTemplate.from_template(
    """Tu es un assistant concis.

Pour répondre à la sous-question suivante, utilise exactement ce format :

Réponse : <une réponse courte>

Étapes:
- <étape 1>
- <étape 2>

Sous-question : {subq}
"""
)

model_answer = ChatOllama(
    model="llama3.2",
    temperature=0.2
)

answer_chain = answer_prompt | model_answer


# ============================================================
# 4. BATCH
# ============================================================

def run_answers(subquestions):

    inputs = [
        {"subq": question}
        for question in subquestions
    ]

    outputs = answer_chain.batch(inputs)

    parsed = []

    for output in outputs:

        text = getattr(
            output,
            "content",
            str(output)
        ).strip()

        answer = None
        steps = []

        for line in text.splitlines():

            if line.lower().startswith("réponse :"):

                answer = line.split(
                    ":",
                    1
                )[1].strip()

            elif re.match(
                r"\s*-\s+",
                line
            ):

                step = re.sub(
                    r"^\s*-\s+",
                    "",
                    line
                ).strip()

                steps.append(step)

        if answer is None:
            answer = text

        if not steps:
            steps = ["Aucune étape analysée."]

        parsed.append(
            {
                "answer": answer,
                "steps": steps,
                "raw": text
            }
        )

    return parsed


run_answers_runnable = RunnableLambda(
    run_answers
)


# ============================================================
# 5. FORMATER LES REPONSES
# ============================================================

def format_subanswers_block(answer_list):

    blocks = []

    for i, answer_data in enumerate(
        answer_list,
        start=1
    ):

        blocks.append(
            f"{i}. Réponse : {answer_data['answer']}"
        )

        blocks.append("   Étapes :")

        for step in answer_data["steps"]:

            blocks.append(
                f"   - {step}"
            )

    return "\n".join(blocks)


format_runnable = RunnableLambda(
    lambda answers: {
        "subanswers_text":
        format_subanswers_block(answers)
    }
)


# ============================================================
# 6. SYNTHESE FINALE
# ============================================================

combine_prompt = PromptTemplate.from_template(
    """Synthétise les sous-réponses suivantes.

Retourne exactement 3 lignes :

1) Réponse finale : <une ligne>
2) Points clés : - <point 1> ; - <point 2>
3) Confiance : <faible/moyenne/élevée>

Sous-réponses :

{subanswers_text}
"""
)

model_combine = ChatOllama(
    model="llama3.2",
    temperature=0
)

combiner = (
    format_runnable
    | combine_prompt
    | model_combine
)


# ============================================================
# 7. PIPELINE LCEL
# ============================================================

pipeline = (
    decomposer
    | parse_subquestions
    | run_answers_runnable
    | combiner
)


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":

    questions = [
        "Comment puis-je réduire la latence dans une application web qui sert des prédictions ML ?",

        "Comment puis-je améliorer la sécurité d'une application web ?"
    ]

    for question in questions:

        print("\n")
        print("=" * 60)
        print("QUESTION")
        print("=" * 60)

        print(question)

        print("\n")
        print("=" * 60)
        print("REPONSE FINALE")
        print("=" * 60)

        result = pipeline.invoke(
            {
                "question": question
            }
        )

        print(
            getattr(
                result,
                "content",
                str(result)
            )
        )
