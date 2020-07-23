from azureml.core.model import Model
# Tip: When model_path is set to a directory, you can use the child_paths parameter to include
#      only some of the files from the directory
ws='Mohith_workspace'
model = Model.register(model_path = "C:/Users/Mohith/Desktop/Datasets/models",
                       model_name = "LogisticRegression_Initial",
                       description = "LogisticRegression model trained outside Azure Machine Learning service",
                       workspace = ws)
