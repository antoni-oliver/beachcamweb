from django.core.management.base import BaseCommand
from django.core.management import call_command
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import download_and_process

class Command(BaseCommand):
    def handle(self, *args, **options):
        scheduler = BlockingScheduler()
        scheduler.add_job(download_and_process.main, CronTrigger(minute=0), next_run_time=datetime.now())
        scheduler.add_job(lambda: call_command('run_pipeline'), CronTrigger(month=1, hour=1, minute=0))
        scheduler.start()