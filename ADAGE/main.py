from pre_defined_functions import *
from target_cols import *
from task_identification import *
from variables import *
from task_to_function import *

def start(query):
    task = task_identifcation(query, available_tasks)
    print('task called')
    target_col = columns_identifier(query)
    print('target_col called')
    print(f'{target_col[0]} {task}: {task_caller(data, task, target_col)}')


query = input("Enter user query: ")
start(query)