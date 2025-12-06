import asyncio
import pandas as pd
from llama_index.core.schema import QueryBundle
from backend import UniversityAgent


QA_DATASET = [
    {
        "query": "What topics are covered in Probability and Statistics?",
        "expected": "BSc_ Probability And Statistics.f23 - IU"
    },
    {
        "query": "Tell me about Data Mining course",
        "expected": "BSc_ Data Mining - IU"
    },
    {
        "query": "Как получить академический отпуск?",
        "expected": "Rules"
    },
    {
        "query": "Grading policy for Mechatronics",
        "expected": "BSc_ Mechatronics - IU"
    },
    {
        "query": "Предметы-пререквизиты для курса практическое машинное обучение и глубокое обучение",
        "expected": "BSc_ Practical Machine Learning Deep Learning - IU"
    },
    {
        "query": "какие цены за общежития",
        "expected": "Расчет_стоимости_размещения_с"
    },
    {
        "query": "Who is Alexander Gasnikov",
        "expected": "Ректор Университета Иннополи"
    },
    {
        "query": "кто руководит университетом Иннополис",
        "expected": "Руководство"
    },
    {
        "query": "Send information about start and end of each semester for bachelors in 2025-2026",
        "expected": "КУГ на 2025-2026 уч.год (программы бакалавриата)"
    },
    {
        "query": "Какие есть штрафы за нарушения ипользования жилой комнаты в общежитии",
        "expected": "Определение_стоимости_ТМЦ"
    },
    {
        "query": "Can I drink alcohol in dormitory",
        "expected": "Приложение к приказу_Правила _"
    },
    {
        "query": "расскажи про размеры стипендии",
        "expected": "Приказ_об_утверждении_размеров_государственных_стипендий"
    },
    {
        "query": "расскажи о программе обмена",
        "expected": "Международный студенческий о"
    },
    {
        "query": "Can I order a cleaning service",
        "expected": "Доп услуги"
    },
    {
        "query": "Tell me about teachers in AI360 in 2024 year",
        "expected": 'преподаватели программы бакалавриата: направление подготовки 09.03.01 Информатика и вычислительная техника, направленность (профиль) образовательной программы "AI360: Инженерия данных", 2024 год набора'
    },
    {
        "query": "Tell me about teachers in AI360",
        "expected": 'преподаватели программы бакалавриата: направление подготовки 09.03.01 Информатика и вычислительная техника, направленность (профиль) образовательной программы "AI360: Инженерия данных", 2024 год набора'
    },
    {
        "query": "Tell me about teachers in masters Software engineering program 2024",
        "expected": 'преподаватели программы магистратуры: направление подготовки 09.04.01 Информатика и вычислительная техника, направленность (профиль) образовательной программы "Программная инженерия", 2024 год набора'
    },
    {
        "query": "How can I contact the university",
        "expected": 'Контакты'
    },
    {
        "query": "Предоставь контакты отдел по работе со студентами",
        "expected": 'Контакты'
    },
    {
        "query": "What will I learn in distributed And network programming course",
        "expected": 'BSc_ Distributed And Network Programming - IU'
    },
    {
        "query": "о чём курс обработка естественного языка",
        "expected": 'BSc_ Natural Language Processing - IU'
    },
    {
        "query": "What documents should I provide when checking into the hostel?",
        "expected": 'Формы_документов_для_размещен'
    },
    {
        "query": "I am looking for internship. What would you recommend for me?",
        "expected": 'Стажировки и работа'
    },
    {
        "query": "What should I know before Optimization?",
        "expected": 'BSc_ Introduction to Optimization.F22 - IU'
    },
    {
        "query": "Зачем нужен курс дискретная математика и логика",
        "expected": 'BSc_ Logic and Discrete Mathematics - IU'
    },
    {
        "query": "Как перейти на бюджетное обучение с платного",
        "expected": 'polozh_splat_nabesplat_02.09.2024'
    },
    {
        "query": "Можно ли курить в общежитии",
        "expected": 'Приложение к приказу_Правила _'
    },
    {
        "query": "Tell me about the rules for using the study room",
        "expected": 'Правила пользования уч.комнат'
    },
    {
        "query": "какй порядок приёма в аспирантуру на 2025 год",
        "expected": 'Правила_приема_аспирантура_2025'
    },
    {
        "query": "по каким дням проходит смена постельного белья во втором корпусе",
        "expected": 'график_смены_белья'
    },
]


async def run_benchmark():
    print("Starting Automated RAG Benchmark...")
    print(f"Test Cases: {len(QA_DATASET)}")

    agent = UniversityAgent()

    base_retriever = agent.index.as_retriever(similarity_top_k=10)

    results = []

    for item in QA_DATASET:
        query_text = item["query"]
        expected_keyword = item["expected"].lower()

        print(f"\nQuestion: {query_text}")

        nodes = await base_retriever.aretrieve(query_text)

        query_bundle = QueryBundle(query_str=query_text)
        reranked_nodes = agent.reranker.postprocess_nodes(nodes, query_bundle)

        top_3_nodes = reranked_nodes[:3]

        found = False
        retrieved_names = []

        for node in top_3_nodes:
            file_name = node.metadata.get("file_name", "Unknown").lower()
            retrieved_names.append(file_name)

            if expected_keyword in file_name:
                found = True

        results.append({
            "Query": query_text,
            "Expected File": item["expected"],
            "Success": 1 if found else 0,
            "Retrieved Top-3": [n[:15] + "..." for n in retrieved_names]  # Обрезаем для красоты
        })

        print(f"--> Found: {found}")


    df = pd.DataFrame(results)

    success_count = df[df["Success"] == 1].shape[0]
    total_count = len(QA_DATASET)
    success_rate = (success_count / total_count) * 100

    print("\n" + "=" * 40)
    print("FINAL BENCHMARK REPORT")
    print("=" * 40)
    pd.set_option('display.max_colwidth', 30)
    print(df[["Query", "Expected File", "Success"]])

    print("\n" + "-" * 40)
    print(f"RETRIEVAL HIT RATE: {success_rate:.1f}%")
    print("-" * 40)

    df.to_csv("benchmark_results.csv", index=False)
    print("Detailed results saved to 'benchmark_results.csv'")


if __name__ == "__main__":
    asyncio.run(run_benchmark())