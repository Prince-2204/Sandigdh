# Generated by Django 4.2.10 on 2024-02-21 07:28

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0003_screenshot'),
    ]

    operations = [
        migrations.RenameField(
            model_name='screenshot',
            old_name='created_at',
            new_name='uploaded_at',
        ),
    ]
