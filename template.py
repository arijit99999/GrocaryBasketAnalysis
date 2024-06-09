import os 
from pathlib import Path
pakage_name='GrocaryBasketAnalysis'
list_of_files=[
    '.github/workflows/.gitkeep',
    f'src/{pakage_name}/__init__.py',
    f'src/{pakage_name}/components/__init__.py',
    f'src/{pakage_name}/components/data_ingestion.py',
    f'src/{pakage_name}/components/data_transformation.py',
    f'src/{pakage_name}/components/model_evaluation.py',
    f'src/{pakage_name}/components/model_trainer.py',
    f'src/{pakage_name}/pipelines/__init__.py',
    f'src/{pakage_name}/pipelines/training_pipeline.py',
    f'src/{pakage_name}/pipelines/prediction_pipeline.py',
    f'src/{pakage_name}/logger.py',
    f'src/{pakage_name}/exception.py',
    f'src/{pakage_name}/utlis/__init__.py',
    f'src/{pakage_name}/utlis/utils.py',
    'notebooks/research.ipynb',
    'notebooks/EDA.ipynb',
     'notebooks/STATS.ipynb',
    'notebooks/data/.gitkeep',
    'requirements.txt',
    'test.py',
    'app.py',
    'setup.py',
    'template/form.html',
    'template/result.html',
    'init_setup.sh',
    'dockerfile',
    'bash.py'
]
for i in list_of_files:
    filepath=Path(i)
    dir,file=os.path.split(filepath)
    if dir!='':
        os.makedirs(dir,exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open (filepath,'w') as f:
            pass
    else:
        print('file already exists')       