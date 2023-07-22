ENV_NAME := universal_da
PYTHON_VERSION := 3.8

create_env:
	@echo "Updating conda..."
	conda update -n base -c defaults conda -y
	@echo "Creating environment..."
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION)
	@echo "Installing python dependencies..."
	conda run -n $(ENV_NAME) torchinstall
	conda run -n $(ENV_NAME) pip install -r requirements.txt
	@echo "Copying directories..."
	cp -r /nas/data/syamagami/GDA/data/GDA_DA_methods/data ./
	mv ./data/Office31 ./data/office
	cp -r /nas/data/syamagami/GDA/data/GDA_DA_methods/Universal-Domain-Adaptation/pretrained_models ./

# remove_envターゲット: condaの環境を削除
remove_env:
	@echo "Removing environment..."
	conda env remove --name $(ENV_NAME)
	@echo "Environment removed successfully."
