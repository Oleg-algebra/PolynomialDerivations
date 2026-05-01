import pandas as pd
import webbrowser

def load_research_results(filename="results_log.jsonl"):
    # 1. Завантаження
    df = pd.read_json(filename, lines=True)

    # 2. Денормалізація (перетворення вкладених ключів у колонки)
    data_df = pd.json_normalize(df['data'])
    final_df = pd.concat([df[['id', 'timestamp']], data_df], axis=1)

    # 3. Конвертація типів (за потреби)
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])

    return final_df



# Використовуємо контекстний менеджер


# Використання:
df = load_research_results()


# 1. Показати всі успішні деривації, де RANK > 1
print(df[df['RANK'] > 1][['id', 'params', 'RANK']])


# 2. Побудувати гістограму часу обчислень
# df['time'].hist(bins=20)

# 3. Перевірити, скільки було знайдено нескінченних множин критичних точок
# (рахуємо, де в списках є тип 'Line of Critical Points')
def has_line(points_list):
    return any(p['type'].startswith('Line') for p in points_list)


df['has_continuum'] = df['critical_points_types'].apply(has_line)
print(f"Кількість континуумів: {df['has_continuum'].sum()}")
print(df.info())