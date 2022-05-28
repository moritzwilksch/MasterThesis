install:
	pip install -r requirements.txt

format:
	black . 
	isort .

backup-db:
	docker exec -t mongo mongodump --db thesis --collection labeled_tweets --gzip -u=$(MONGO_USER) -p=$(MONGO_PASSWD) --authenticationDatabase=thesis --archive > labeled_tweets_backup.gz
	python src/scripts/upload_backup.py

