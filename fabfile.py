from __future__ import print_function, unicode_literals

import io
import os
import sys
from datetime import datetime

from fabric import task

from getpass import getpass
from posixpath import join

from invoke import Collection
from paramiko import SSHConfig

################
# Config setup #
################

from production.settings import FABRIC as conf

main_app = conf.MAIN_APP

secret_key = conf.SECRET_KEY
db_pwd = conf.DB_PWD or getpass("Enter the database password: ")
github_external_repo = conf.GITHUB.REPO_PATH
github_auth_token = conf.GITHUB.AUTH_TOKEN
github_user = conf.GITHUB.USER or getpass("Enter the git user: ")

servers = conf.SERVERS
hosts = [f"{s['USER']}@{s['IP']}" for s in servers]

ssh_user = servers[0]['USER']
ssh_pwd = servers[0]['PWD']

domains = conf.DOMAINS
superuser_name = conf.SUPERUSER.NAME
superuser_email = conf.SUPERUSER.EMAIL
superuser_pwd = conf.SUPERUSER.PWD

proj_name = conf.PROJECT_NAME
venv_home = f"/home/{ssh_user}/venvs"
reqs_path = conf.REQUIREMENTS_PATH
locale = conf.LOCALE.replace("UTF-8", "utf8")

gunicorn_num_workers = conf.GUNICORN.NUM_WORKERS

venv_path = join(venv_home, proj_name)
proj_path = f"/home/{ssh_user}/code/{proj_name}"
logs_home = f"/home/{ssh_user}/logs/{proj_name}"

##################
# Template setup #
##################

# Each template gets uploaded at production time, and their reload command are run.

templates = {
    "nginx": {
        "local_path": "production/templates/nginx.conf.template",
        "remote_path": f"/etc/nginx/sites-enabled/{proj_name}.conf",
        "reload_commands": ["nginx -t",
                            "service nginx restart"],
    },
    "supervisor": {
        "local_path": "production/templates/supervisor.conf.template",
        "remote_path": f"/etc/supervisor/conf.d/{proj_name}.conf",
        "reload_commands": [f"supervisorctl update gunicorn_{proj_name}"],
    },
    "gunicorn": {
        "local_path": "production/templates/gunicorn.conf.py.template",
        "remote_path": f"{proj_path}/gunicorn.conf.py",
    },
    "settings": {
        "local_path": "production/templates/local_settings.py.template",
        "remote_path": f"{proj_path}/{main_app}/local_settings.py",
    },
    "ssh_config": {
        "local_path": "production/templates/ssh_config.template",
    },
    "secrets": {
        "local_path": "production/secrets.py",
        "remote_path": f"{proj_path}/production/secrets.py",
        "noformat": True,
    },
}

format_templates = {  # Dict with project-dependant strings to format templates
    'secret_key': secret_key,
    'proj_path': proj_path,
    'proj_name': proj_name,
    'main_app': main_app,
    'venv_path': venv_path,
    'logs_home': logs_home,
    'ssh_user': ssh_user,
    'db_pwd': db_pwd,
    'locale': locale,
    'gunicorn_num_workers': gunicorn_num_workers,
    'domains_nginx': " ".join(domains),
    'domains_python': ", ".join([f"'{s}'" for s in domains]),
    'domains_regex': "|".join(domains),
    'django_admins': f"('{superuser_name}', '{superuser_email}')",
    'FABRIC': conf,
}

prefix_virtualenv = f'source {venv_path}/bin/activate && cd {proj_path} && '
prefix_manage = f'{prefix_virtualenv} python3.10 manage.py '


def run_local(connection, cmd):
    connection.local(cmd, echo=True)

@task(hosts=hosts)
def prepare_deploy(c):
    path_activate = os.path.join(os.path.split(sys.executable)[0], 'activate')
    run_local(c, f"source {path_activate} && python manage.py makemigrations")
    c.local(f"source {path_activate} && pip freeze > {reqs_path}")
    c.local(f"source {path_activate} && git add -A", warn=True)
    c.local(f"source {path_activate} && git commit -m 'fabfile'", warn=True)
    c.local(f"source {path_activate} && git push", warn=True)


@task(hosts=hosts)
def deploy(c, prepare=True):
    # if not exists(proj_path):
    #     return RuntimeError('Project does not exist, initialize with `fab create`.')

    if prepare:
        prepare_deploy(c)

    c.run(f"{prefix_virtualenv} git pull origin main")
    c.run(f"{prefix_virtualenv} python3.10 -m pip install -r {proj_path}/{reqs_path}")

    updatetemplates(c)
    c.run(f"{prefix_manage} collectstatic -v 0 --noinput")
    c.run(f"{prefix_manage} migrate --noinput")

    return restart(c)


# noinspection SqlNoDataSourceInspection,SqlResolve
@task(hosts=hosts)
def create(c, prepare_before_deploying=True):
    # Generate project locale
    if locale not in c.run("locale -a", hide=True).stdout:
        c.sudo(f"locale-gen {locale}")
        c.sudo(f"update-locale {locale}")
        c.sudo("service postgresql restart_gunicorn")
        c.run("exit")

    c.run(f"mkdir -p {logs_home}")
    c.run(f"mkdir -p {venv_home}")
    c.run(f"mkdir -p {proj_path}")

    # Create virtual env
    c.run(f"cd {venv_home} && virtualenv -p python3  {proj_name}")
    c.run(f"{prefix_virtualenv} curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10")
    c.run(f"{prefix_virtualenv} python3.10 -m pip install gunicorn setproctitle psycopg2-binary django-compressor python-memcached")

    # Create DB and DB user
    db_pwd_sanitized = db_pwd.replace("'", "\'")
    run_psql(c,
             f"CREATE USER {proj_name} WITH ENCRYPTED PASSWORD '{db_pwd_sanitized}';")
    run_psql(c,
             f"CREATE DATABASE {proj_name} WITH OWNER {proj_name} ENCODING = 'UTF8' "
             f"LC_CTYPE = '{locale}' LC_COLLATE = '{locale}' TEMPLATE template0;")

    # Initialize project git with GitHub authentication token
    c.run(f"mkdir -p {proj_path}")
    c.run(f"{prefix_virtualenv} git init")
    c.run(
        f"{prefix_virtualenv} git remote add origin https://github.com/{github_external_repo}")
    c.run(f"{prefix_virtualenv} git remote -v")  # Verify

    # Deploy
    deploy(c, prepare=prepare_before_deploying)

    # Start gunicorn service
    c.sudo(f"supervisorctl start gunicorn_{proj_name}")

    # Bootstrap the DB
    addsuperuser(c)

    # Configure rabbitmq vhost and celery
    c.sudo(f"rabbitmqctl add_vhost vhost-{proj_name}", warn=True)
    c.sudo(f"rabbitmqctl add_user '{proj_name}' '{proj_name}'", warn=True)  # Pwd = {proj_name}
    c.sudo(f"rabbitmqctl set_permissions -p 'vhost-{proj_name}' '{proj_name}' '.*' '.*' '.*'", warn=True)

    return True


@task(hosts=hosts)
def remove(c):
    """
    Blow away the current project.
    """
    c.run(f"rm -rf {venv_path}", warn=True)
    c.run(f"rm -rf {proj_path}", warn=True)
    for template in templates.values():
        if 'remote_path' in template:
            c.sudo(f"rm {template['remote_path']}", warn=True)

    c.sudo("supervisorctl update")
    run_psql(c, f"DROP DATABASE IF EXISTS {proj_name};")
    run_psql(c, f"DROP USER IF EXISTS {proj_name};")


##########
# Utils  #
##########

@task(hosts=hosts)
def stop(c):
    """
    Restart gunicorn worker processes for the project.
    If the processes are not running, they will be started.
    """
    c.run(f"kill -HUP `cat {proj_path}/gunicorn.pid`", warn=True)
    c.sudo(f"supervisorctl stop gunicorn_{proj_name} celerybeat_{proj_name} celeryworker_{proj_name}", warn=True)


@task(hosts=hosts)
def restart(c):
    """
    Restart gunicorn worker processes for the project.
    If the processes are not running, they will be started.
    """
    c.run(f"kill -HUP `cat {proj_path}/gunicorn.pid`", warn=True)
    c.sudo("supervisorctl reread", warn=True)
    c.sudo(f"supervisorctl restart gunicorn_{proj_name} celerybeat_{proj_name} celeryworker_{proj_name}", warn=True)


@task(hosts=hosts)
def run(c, cmd=None):
    if cmd is None:
        print('Interactive (remote) run. (Press enter to end)')
        while True:
            cmd = input('> ')
            if cmd == '':
                return
            try:
                run(c, cmd)
            except:
                print('[Error] (Press enter to end)')
    # noinspection PyPep8
    sanitized_cmd = cmd.replace("`", "\\\`")
    c.run(f'{prefix_virtualenv} {sanitized_cmd}')


@task(hosts=hosts)
def run_psql(c, sql, hide=False):
    """
    Runs SQL against the project's database.
    """
    out = c.sudo(f'psql -c "{sql}"', hide=hide, user="postgres")
    if hide:
        print(sql)
    return out


@task(hosts=hosts)
def django(c, code):
    """
    Runs Python code in the project's virtual environment, with Django loaded.
    """
    setup = (f"import os;"
             f"os.environ['DJANGO_SETTINGS_MODULE']='{main_app}.settings';"
             f"import django;"
             f"django.setup();")
    # noinspection PyPep8
    sanitized_code = code.replace("`", "\\\`")
    return c.run(f'{prefix_virtualenv} python3.10 -c "{setup}{sanitized_code}"')


@task(hosts=hosts)
def logs(c):
    options = ['Quit'] + c.run(f"ls {logs_home}/", hide=True).stdout.split()
    selection = ""
    while True:
        if selection == "":
            print('List of log files:')
            for idx, filename in enumerate(options):
                print(f"{idx} - {filename}")
            selection = input(f"Please select (0-{len(options) - 1}): ")
        else:
            selection = input(f"Please select (0-{len(options) - 1}, or entry for list of logs): ")
        try:
            idx_s = int(selection)
            if idx_s == 0:
                return
            else:
                c.run(f"tail -n 3000 {logs_home}/{options[idx_s]}")
        except BaseException:
            pass


@task(hosts=hosts)
def removelogs(c):
    c.run(f"rm -f {logs_home}/*.log*", hide=True)
    restart(c)


@task(hosts=hosts)
def addsuperuser(c):
    if superuser_pwd and superuser_name:
        django(
            c,
            f"from django.contrib.auth import get_user_model;"
            f"User = get_user_model();"
            f"u, _ = User.objects.get_or_create(username='{superuser_name}', email='{superuser_email}');"
            f"u.is_staff = u.is_admin = u.is_superuser = True;"
            f"u.set_password('{superuser_pwd}');"
            f"u.save();")


@task(hosts=hosts)
def updatetemplates(c):
    for name, template_info in templates.items():
        if 'remote_path' in template_info:
            with open(template_info['local_path'], 'r') as template_file:
                print(f'\n# Update template: {name}\n###\n')
                if 'noformat' in template_info:
                    template_as_str = template_file.read()
                else:
                    template_as_str = template_file.read().format(**format_templates)
                upload_file(c, io.StringIO(template_as_str), template_info['remote_path'])
                if 'reload_commands' in template_info:
                    for cmd in template_info['reload_commands']:
                        c.sudo(cmd, warn=True)


def upload_file(c, file_object, remote_path):
    c.put(file_object, 'fabupload.aux')
    c.sudo(f'mv /home/{ssh_user}/fabupload.aux {remote_path}', warn=True)


@task(hosts=hosts)
def replicatedatabase(c):
    dump_fname = f'db_dump_{datetime.now().strftime("%Y%M%d_%H%M%S")}'
    c.run(
        f'{prefix_virtualenv} python3.10 manage.py dumpdata --format json --indent 4 --natural-foreign -e contenttypes.ContentType -e auth.Permission --output {dump_fname}.json')
    c.local(f'rsync {hosts[0]}:{proj_path}/{dump_fname}.json {dump_fname}.json')
    c.local(f'rsync -a {hosts[0]}:{proj_path}/static/media/* static/media/')
    c.run(f'{prefix_virtualenv} rm {dump_fname}.json')

    c.local(f'python3.10 manage.py flush')
    c.local(f'python3.10 manage.py loaddata {dump_fname}.json')


################################
# Configure fabric tasks (CLI) #
################################


with open(templates['ssh_config']['local_path'], 'r') as file:
    template_str = file.read()
    ssh_config_chunks = [
        template_str.format(
            name=s['NAME'],
            ip=s['IP'],
            user=s['USER'],
            keyfile=s['KEYFILE_LOCALPATH'],
        ) for s in servers
    ]
    config = SSHConfig.from_text("\n\n".join(ssh_config_chunks))

ns = Collection(prepare_deploy, deploy, create, run, run_psql,
                django, remove, updatetemplates, restart, stop,
                logs, removelogs, replicatedatabase, addsuperuser)
ns.configure({
    'sudo': {
        'password': ssh_pwd,
    },
    'connect_kwargs': {
        'password': ssh_pwd
    },
})
