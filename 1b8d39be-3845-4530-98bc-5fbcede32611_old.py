#!/usr/bin/env python
# coding: utf-8

# # Задание 1. Проведите исследование
# 
#     Откроем файл с данными (pr_data) со статистикой о платёжеспособности банковских клиентов
#  
#     Задача. Влияет ли семейное положение и количество детей клиента на факт возврата кредита в срок?
#     
#     Описание данных
# 
# - children — количество детей в семье
# - days_employed — трудовой стаж в днях
# - dob_years — возраст клиента в годах
# - education — образование клиента
# - education_id — идентификатор образования
# - family_status — семейное положение
# - family_status_id — идентификатор семейного положения
# - gender — пол клиента
# - income_type — тип занятости
# - debt — имел ли задолженность по возврату кредитов
# - total_income — доход в месяц

# In[5]:


# Импортируем библиотеку pandas и прочитаем файл с исходными данными
import pandas as pd
pr_data = pd.read_csv('/datasets/data.csv')


#     Изучим общую информацию о данных, используя info() метод.

# In[6]:


pr_data.info()
pr_data.head(20)


# Общее количество значений в данных 21525. В двух параметрах значений существенно меньше: опыт работы и доход.

# In[7]:


print(pr_data.isnull().sum())

data_null = pr_data[pr_data["days_employed"].isnull()]
'Значения отсутствуют в: {:.1%}' .format(len(data_null) / len(pr_data))


# Отсутствует количественные значения data_employed и total_income в 10% от общих данных. Рассмотрим уникальные значения данных для каждого столбца.

# In[8]:


for row in pr_data: 
  print(pr_data[row].value_counts())


# Выводы:
# Столбец "children". Ошибочные значения "20", -1. Поменять на 2 и 1 соответственно
# Столбец "gender". Строки с пропущенными данными.
# Столбец "education". Дубликаты в названиях. Привести к одному регистру.
# Столбец "dob_days". Нулевые значения.
# Столбец "purpose". Много дубликатов с похожими или идентичными причинами. Требуется объеденить в одинаковые значения.
# Столбец "days_employed". Положительные и отрицательные значения. Значения могут быть только Integer.
# Столбец "total_income". Пропущенные значения (2174)
# Столбец "days_employed". Пропущенные значения (2174)

# In[9]:


### Столбец Children
pr_data['children'] = pr_data['children'].replace(20, 2)
pr_data['children'] = pr_data['children'].replace(-1, 1)

###  Столбец Gender
# Узнаем количество пропущенных значений
pr_data[pr_data['gender'] == 'XNA'].count()

# Строка с пропущенным значением всего одна. Ее можно исключить из выборки. 
pr_data = pr_data[pr_data['gender'] != 'XNA']

### Столбец dob_years. 101 пропущенное значение. 
pr_data['dob_years'].min()
pr_data['dob_years'].max()

### Столбец education 
# Приведем все значения к одному регистру
pr_data['education'] = pr_data['education'].str.lower()
# В столбце education теперь 5 уникальных значений
pr_data['education'].unique()


# In[10]:


# Для того чтобы заполнить пропущенные значения посчитаем медиану возраста клиента в годах (dob_years) для каждого типа занятости

income_type_gr = pr_data.groupby('income_type')['dob_years'].median()
income_type_gr 


# In[12]:


# pr_data.is_copy = False

# Заполним пропущенные значения в столбце "dob_years" на основании медианы для каждого типа дохода
pr_data.loc[(pr_data['income_type'] == 'пенсионер') & (pr_data['dob_years'] == 0), 'dob_years'] = 60
pr_data.loc[(pr_data['income_type'] == 'студент') & (pr_data['dob_years'] == 0), 'dob_years'] = 22
pr_data.loc[(pr_data['income_type'] == 'сотрудник') & (pr_data['dob_years'] == 0), 'dob_years'] = 39
pr_data.loc[(pr_data['income_type'] == 'предприниматель') & (pr_data['dob_years'] == 0), 'dob_years'] = 42.5
pr_data.loc[(pr_data['income_type'] == 'компаньон') & (pr_data['dob_years'] == 0), 'dob_years'] = 39
pr_data.loc[(pr_data['income_type'] == 'госслужащий') & (pr_data['dob_years'] == 0), 'dob_years'] = 40
pr_data.loc[(pr_data['income_type'] == 'в декрете') & (pr_data['dob_years'] == 0), 'dob_years'] = 39
pr_data.loc[(pr_data['income_type'] == 'безработный ') & (pr_data['dob_years'] == 0), 'dob_years'] = 38

# Сделаем проверку 
pr_data['dob_years'].isnull().sum()


# In[13]:


# Сделаем графическую проверку данных. Данные теперь распределны от 19 до 75 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x = pr_data['dob_years']
plt.hist(x, bins=20)
plt.ylabel('No of times')
plt.show()


# In[14]:


### Чтобы заполнить пропущенные значения в столбце трудовой стаж (days_employed) в днях можно взять медиану для каждой группы возраста. Для этого создадим новый столбец "age_group"
# 1 группа: 19 - 30
# 2 группа: 30 - 40
# 3  группа: 40 - 55 
# 4  группа: 55 - 75

def days_employed(row):
    
    age = row['dob_years']

    if age <= 30:
        return '1 группа'
    
    if age <= 40 and age > 30:
        return '2 группа'
    
    if age <= 55 and age > 40:
        return '3 группа'
    
    if age <= 80 and age > 55:
        return '4 группа'

pr_data['age_group'] = pr_data.apply(days_employed, axis=1)

pr_data['days_employed'] = pr_data['days_employed'].fillna(0)
pr_data.isnull().sum()

pr_data.groupby('age_group')['days_employed'].mean()[0]

pr_data.loc[(pr_data['age_group'] == '1 группа') & (pr_data['days_employed'] == 0 ), 'days_employed'] = pr_data.groupby('age_group')['days_employed'].mean()[0]
pr_data.loc[(pr_data['age_group'] == '2 группа') & (pr_data['days_employed'] == 0 ), 'days_employed'] = pr_data.groupby('age_group')['days_employed'].mean()[1]
pr_data.loc[(pr_data['age_group'] == '3 группа') & (pr_data['days_employed'] == 0 ), 'days_employed'] = pr_data.groupby('age_group')['days_employed'].mean()[2]
pr_data.loc[(pr_data['age_group'] == '4 группа') & (pr_data['days_employed'] == 0 ), 'days_employed'] = pr_data.groupby('age_group')['days_employed'].mean()[3]

pr_data[pr_data['total_income'] == 0].count()


# In[15]:


# Чтобы заполнить пропущенные значения в столбце Total Income можно использовать медиану дохода для каждого типа дохода (income_type)

pr_data['total_income'] = pr_data['total_income'].fillna(0)

pr_data.groupby('income_type')['total_income'].median()

pr_data.loc[(pr_data['total_income'] == 0) & (pr_data['income_type'] == 'безработный'), 'total_income'] = pr_data.groupby('income_type')['total_income'].median()[0]
pr_data.loc[(pr_data['total_income'] == 0) & (pr_data['income_type'] == 'в декрете'), 'total_income'] = pr_data.groupby('income_type')['total_income'].median()[1]
pr_data.loc[(pr_data['total_income'] == 0) & (pr_data['income_type'] == 'госслужащий'), 'total_income'] = pr_data.groupby('income_type')['total_income'].median()[2]
pr_data.loc[(pr_data['total_income'] == 0) & (pr_data['income_type'] == 'компаньон'), 'total_income'] = pr_data.groupby('income_type')['total_income'].median()[3]
pr_data.loc[(pr_data['total_income'] == 0) & (pr_data['income_type'] == 'пенсионер'), 'total_income'] = pr_data.groupby('income_type')['total_income'].median()[4]
pr_data.loc[(pr_data['total_income'] == 0) & (pr_data['income_type'] == 'предприниматель'), 'total_income'] = pr_data.groupby('income_type')['total_income'].median()[5]
pr_data.loc[(pr_data['total_income'] == 0) & (pr_data['income_type'] == 'сотрудник'), 'total_income'] = pr_data.groupby('income_type')['total_income'].median()[6]
pr_data.loc[(pr_data['total_income'] == 0) & (pr_data['income_type'] == 'студент'), 'total_income'] = pr_data.groupby('income_type')['total_income'].median()[7]


# In[16]:


# Проверим новые данные для дохода в месяц
x = abs(pr_data['total_income'])
plt.hist(x, bins=60)
plt.ylabel('No of times')
plt.show()


# In[17]:


# Сгрупируем данные о доходе в месяц 
# 1 группа: 15 - 55
# 2 группа: 55 - 120
# 3  группа: 120 - 200
# 4  группа: 200 - 2000000

def income_group(row):
    
    income = row['total_income']

    if income <= 55000:
        return 'Доход 15-55'
    
    if income <= 120000 and income > 55000:
        return 'Доход 55-20'
    
    if income <= 200000 and income > 120000:
        return 'Доход 120-200'
    
    if income <= 4000000 and income > 200000:
        return 'Доход > 200'

pr_data['income_gr'] = pr_data.apply(income_group, axis=1)

pr_data


# ## Дубликаты

# In[18]:


# Посчитаем количество дубликатов в наборе данных. Дубликаты в категориях были исключены выше для каждого столбца. 
pr_data.duplicated().sum()

# Удалим дубликаты используя метод drop_duplicates() с новой индексацией
pr_data = pr_data.drop_duplicates().reset_index(drop=True)


# ## Изменить тип данных

# In[19]:


# Изучим текущие типы данных 
pr_data.info()


# In[20]:


# Для изменения типа данных используем метод astype() так как нам нужно изменить float на int64
# Изменим значения столбца total_income в числовой тип int64 (целое число) 
pr_data['total_income'] = pr_data['total_income'].astype('int')

# Изменим значения столбца dob_years в числовой тип int64 (целое число) 
pr_data['dob_years'] = pr_data['dob_years'].astype('int')

# Изменим значения столбца days_employed в числовой тип int64 (целое число) и возьмем значения по модулю 
pr_data['days_employed'] = pr_data['days_employed'].astype('int')
pr_data['days_employed'] = abs(pr_data['days_employed'])


# ## Лемматизация целей кредита

# In[21]:


# Импортируем библиотеку с функцией лемматизации на русском языке — pymystem3
import json
from pymystem3 import Mystem
from collections import Counter
m = Mystem()


# In[22]:


# Создадим функцию которая возвращает строку с ключевыми словами цели кредита
def lemma_pur(purpose):
    lemma = ' ' .join(m.lemmatize(purpose))
    return lemma

pr_data['purpose_word'] = pr_data['purpose'].apply(lemma_pur)

# Отобразим уникальные значения (цели кредита)
pr_data['purpose_word'].unique()


# In[23]:


# Создадим новый столбец (purpose_cat) с категориями на основании результатов лемматизации
def purpose_cat(list):
    if 'автомобиль' in list:
        return "автомобиль"
    if "образование" in list:
        return "образование"
    if "свадьба" in list:
        return "свадьба"
    if "недвижимость" in list:
        return "недвижимость"
    if "строительство" in list:
        return "строительство"
    if "жилье" in list:
        return "жилье"

pr_data['purpose_cat'] = pr_data['purpose_word'].apply(purpose_cat)

# После группировки цели кредита разделены на 6 групп (автомобиль, образование, свадьба, недвижимость, строительство, жилье)
pr_data


# ## Задание 2. Опишите свои выводы
# - Данные очищенны. Теперь можно сгрупировать их для того чтобы ответить на поставленые вопросы в задании

# In[24]:


late_payment = pr_data[(pr_data['debt'] == 1)].count() / pr_data[(pr_data['debt'] == 0)].count()
late_payment['debt']


# в 8,8% случаем кредит не выплачивается вовремя. Посмотрим как меняется это значение в зависимости от других факторов.

# ### Есть ли зависимость между наличием детей и возвратом кредита в срок?

# In[25]:


data_pivot1 = pr_data.pivot_table(index=['debt'], columns='children', values = 'family_status_id', aggfunc='count')
ch0 = data_pivot1[0][1] / data_pivot1[0][0]
ch1 = data_pivot1[1][1] / data_pivot1[1][0]
ch2 = data_pivot1[2][1] / data_pivot1[2][0]
ch3 = data_pivot1[3][1] / data_pivot1[3][0]
ch4 = data_pivot1[4][1] / data_pivot1[4][0]

print("{0:.2f}% Нет детей".format(ch0*100))
print("{0:.2f}% 1 Ребенок".format(ch1*100))
print("{0:.2f}% 2 Ребенка".format(ch2*100))
print("{0:.2f}% 3 Ребенка".format(ch3*100))
print("{0:.2f}% 4 Ребенка".format(ch4*100))


# В среднем появляние ребенка увеличивает возможность просрочки платежа по кредиту. 

# ### Есть ли зависимость между семейным положением и возвратом кредита в срок?

# In[26]:


data_pivot2 = pr_data.pivot_table(index=['debt'], columns='family_status', values='age_group', aggfunc='count')
no_family = data_pivot2['Не женат / не замужем'][1] / data_pivot2['Не женат / не замужем'][0]
divorce = data_pivot2['в разводе'][1] / data_pivot2['в разводе'][0]
widow = data_pivot2['вдовец / вдова'][1] / data_pivot2['вдовец / вдова'][0]
partner = data_pivot2['гражданский брак'][1] / data_pivot2['гражданский брак'][0]
family = data_pivot2['женат / замужем'][1] / data_pivot2['женат / замужем'][0]
print("{0:.2f}% Не женат / не замужем".format(no_family*100))
print("{0:.2f}% в разводе".format(divorce*100))
print("{0:.2f}% вдовец / вдова".format(widow*100))
print("{0:.2f}% гражданский брак".format(partner*100))
print("{0:.2f}% женат / замужем".format(family*100))


# - Не женатые/ Не замужние или люди в гражданском браке реже выплачивают кредит в срок.

# ### Есть ли зависимость между уровнем дохода и возвратом кредита в срок?

# In[27]:


data_pivot3 = pr_data.pivot_table(index=['debt'], columns='income_gr', values='age_group', aggfunc='count')

in1 = data_pivot3['Доход 120-200'][1] / data_pivot3['Доход 120-200'][0]
in2 = data_pivot3['Доход 15-55'][1] / data_pivot3['Доход 15-55'][0]
in3 = data_pivot3['Доход 55-20'][1] / data_pivot3['Доход 55-20'][0]
in4 = data_pivot3['Доход > 200'][1] / data_pivot3['Доход > 200'][0]

print("{0:.2f}% Доход 120-200".format(in1*100))
print("{0:.2f}% Доход 15-55".format(in2*100))
print("{0:.2f}% Доход 55-20".format(in3*100))
print("{0:.2f}% Доход > 200".format(in4*100))


# Парадоксально люди с наименьшим доходом чаще других групп выплачивают кредит в срок.

# ### Как разные цели кредита влияют на его возврат в срок?

# In[28]:


data_pivot4 = pr_data.pivot_table(index=['debt'], columns='purpose_cat', values='age_group', aggfunc='count')
data_pivot4

in1 = data_pivot4['автомобиль'][1] / data_pivot4['автомобиль'][0]
in2 = data_pivot4['жилье'][1] / data_pivot4['жилье'][0]
in3 = data_pivot4['недвижимость'][1] / data_pivot4['недвижимость'][0]
in4 = data_pivot4['образование'][1] / data_pivot4['образование'][0]
in5 = data_pivot4['свадьба'][1] / data_pivot4['свадьба'][0]

print("{0:.2f}% автомобиль".format(in1*100))
print("{0:.2f}% жилье".format(in2*100))
print("{0:.2f}% недвижимость".format(in3*100))
print("{0:.2f}% образование".format(in4*100))
print("{0:.2f}% свадьба".format(in5*100))


# Люди с большей вероятностью выплачивают кредит если он взять на операции с жильем или недвижимостью или на свадьбу. 

#  ## Общий вывод

# При работе с данными были заполнены пропущенные значения, удалены дубликаты и выделены категории для целей кредита, группы по возрасту и образованию. Применив сводные таблицы, я оценил, какие критерии влияют на возврат кредита. Дополнительный анализ может помочь более точно предсказать вероятность вылпаты кредита в срок.

# In[ ]:




