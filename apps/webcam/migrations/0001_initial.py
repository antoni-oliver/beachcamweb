# Generated by Django 5.0.1 on 2025-04-10 21:30

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='ImageProvider',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('url', models.CharField(help_text="Can use `%Y`, `%m`, `%d`, etc. as in python's strftime(...)", max_length=2048)),
            ],
        ),
        migrations.CreateModel(
            name='StreamProvider',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('url', models.CharField(help_text='Use .m3u8 or .html that generates a .m3u8 after clicking on a html element.', max_length=2048)),
                ('clickable_element_xpath', models.CharField(max_length=2048)),
                ('seconds', models.IntegerField(blank=True, default=10, help_text='Seconds to record for the video')),
            ],
        ),
        migrations.CreateModel(
            name='YoutubeProvider',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('url', models.CharField(help_text='Only for https://www.youtube.com/watch?v=(...) links.', max_length=2048)),
                ('seconds', models.IntegerField(blank=True, default=10, help_text='Seconds to record for the video')),
            ],
        ),
        migrations.CreateModel(
            name='WebCam',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('beach_name', models.CharField(max_length=200, unique=True)),
                ('slug', models.CharField(blank=True, max_length=200, unique=True)),
                ('beach_latitude', models.DecimalField(blank=True, decimal_places=6, max_digits=9, null=True)),
                ('beach_longitude', models.DecimalField(blank=True, decimal_places=6, max_digits=9, null=True)),
                ('description', models.CharField(blank=True, max_length=255, null=True)),
                ('available', models.BooleanField(default=True)),
                ('probe_freq_mins', models.IntegerField(default=60)),
                ('max_crowd_count', models.IntegerField(default=0)),
                ('object_id', models.PositiveIntegerField()),
                ('content_type', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='contenttypes.contenttype')),
            ],
        ),
    ]
