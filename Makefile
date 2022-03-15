install:
	pip install -r requirements.txt

format:
	black . 
	isort .