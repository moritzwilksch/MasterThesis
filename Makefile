install:
	pip install -r requirements.txt

format:
	black . 
	isort .

backup-db:
	docker exec -t mongo mongodump --db thesis --gzip -u=$(MONGO_USER) -p=$(MONGO_PASSWD) --authenticationDatabase=thesis --archive > backup.gz
	python src/utils/upload_backup.py

