install:
	pip install -r requirements.txt

format:
	black . 
	isort .

__backup:
	docker exec -t mongo mongodump --db thesis --collection labeled_tweets --gzip -u=$(MONGO_USER) -p=$(MONGO_PASSWD) --authenticationDatabase=thesis --archive > labeled_tweets_backup.gz
	# python src/scripts/upload_backup.py

label:
	python src/labeling/labeling_tool.py

dashboard:
	optuna-dashboard "sqlite:///tuning/dl_optuna.db"

backup:
	python src/scripts/upload_backup.py

tensorboard:
	tensorboard --logdir lightning_logs/

rm-dl-artifacts:
	-rm -rf lightning_logs/
	-rm outputs/tokenizers/*