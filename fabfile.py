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
from termcolor import colored

################
# Config setup #
################

from deployment.conf import DEPLOYMENT_CONF as CONF

main_app = CONF.MAIN_APP

secret_key = CONF.SECRET_KEY
db_pwd = CONF.DB_PWD or getpass("Enter the database password: ")
github_external_repo = CONF.GITHUB.REPO_PATH
github_auth_token = CONF.GITHUB.AUTH_TOKEN
github_user = CONF.GITHUB.USER or getpass("Enter the git user: ")

servers = CONF.SERVERS
hosts = [f"{s['USER']}@{s['IP']}" for s in servers]

ssh_user = servers[0]['USER']
ssh_pwd = servers[0]['PWD']

domains = CONF.DOMAINS
superuser_name = CONF.SUPERUSER.NAME
superuser_email = CONF.SUPERUSER.EMAIL
superuser_pwd = CONF.SUPERUSER.PWD

proj_name = CONF.PROJECT_NAME
venv_home = f"/home/{ssh_user}/venvs"
reqs_path = CONF.REQUIREMENTS_PATH
locale = CONF.LOCALE.replace("UTF-8", "utf8")

gunicorn_num_workers = CONF.GUNICORN.NUM_WORKERS

venv_path = join(venv_home, proj_name)
proj_path = f"/home/{ssh_user}/code/{proj_name}"
logs_home = f"/home/{ssh_user}/logs/{proj_name}"

##################
# Template setup #
##################

# Each template gets uploaded at production time, and their reload command are run.

templates = {
    "nginx": {
        "local_path": "deployment/templates/nginx.conf.template",
        "remote_path": f"/etc/nginx/sites-enabled/{proj_name}.conf",
        "reload_commands": ["nginx -t",
                            "service nginx restart"],
    },
    "supervisor": {
        "local_path": "deployment/templates/supervisor.conf.template",
        "remote_path": f"/etc/supervisor/conf.d/{proj_name}.conf",
        "reload_commands": [f"supervisorctl update gunicorn_{proj_name}"],
    },
    "gunicorn": {
        "local_path": "deployment/templates/gunicorn.conf.py.template",
        "remote_path": f"{proj_path}/gunicorn.conf.py",
    },
    "settings": {
        "local_path": "deployment/templates/production_settings.py.template",
        "remote_path": f"{proj_path}/{main_app}/production_settings.py",
    },
    "ssh_config": {
        "local_path": "deployment/templates/ssh_config.template",
    },
    "crontab": {
        "local_path": "deployment/templates/crontab.template",
        "remote_path": f"/etc/cron.d/{proj_name}",
        "reload_commands": [f"chown root /etc/cron.d/{proj_name}",
                            f"chmod 600 /etc/cron.d/{proj_name}"],
    },
    "secrets": {
        "local_path": "deployment/secrets.py",
        "remote_path": f"{proj_path}/deployment/secrets.py",
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
    'FABRIC': CONF,
}


path_local_activate = os.path.join(os.path.split(sys.executable)[0], 'activate')
django_setup = (
    f"import os;"
    f"os.environ['DJANGO_SETTINGS_MODULE']='{main_app}.settings';"
    f"import django;"
    f"django.setup();"
)


def print_task_header(task_name):
    print(colored(f"{'=' * 40}\n    TASK: {task_name}\n{'=' * 40}", 'yellow'))

def local_virtualenv(connection, cmd, echo=True, **kwargs):
    if echo:
        print(colored(f"Local VEnv >> {cmd}", 'cyan'))
    return connection.local(f"source {path_local_activate} && {cmd}", **kwargs)


def remote_shell(connection, cmd, echo=True, **kwargs):
    if echo:
        print(colored(f"Remote Shell >> {cmd}", 'blue'))
    return connection.run(cmd, **kwargs)


def remote_sudo(connection, cmd, echo=True, **kwargs):
    if echo:
        print(colored(f"Remote Sudo  >> {cmd}", 'red'))
    return connection.sudo(cmd, **kwargs)


def remote_sql(connection, cmd, echo=True, user="postgres", **kwargs):
    if echo:
        print(colored(f"Remote PSQL >> {cmd}", 'light_red'))
    return remote_sudo(connection, f"psql -c \"{cmd}\"", echo=False, user=user, **kwargs)


def remote_virtualenv(connection, cmd, echo=True, **kwargs):
    if echo:
        print(colored(f"Remote VEnv  >> {cmd}", 'dark_grey'))
    return remote_shell(connection, f'source {venv_path}/bin/activate && cd {proj_path} && {cmd}', echo=False, **kwargs)


def remote_python(connection, cmd, echo=True, **kwargs):
    if echo:
        print(colored(f"Remote python >> {cmd}", 'green'))
    return remote_virtualenv(connection, f"python3.10 {cmd}", echo=False, **kwargs)


def remote_django(connection, cmd, echo=True, **kwargs):
    if echo:
        print(colored(f"Remote Django >> {cmd}", 'magenta'))
    sanitized_cmd = cmd.replace("`", r"\`")
    return remote_python(connection, f"{django_setup} {sanitized_cmd}", echo=False, **kwargs)


def upload_file(c, file_object, remote_path, echo=True):
    if echo:
        print(colored(f"UPLOAD/UPDATE FILE >> {remote_path}", 'cyan'))
    c.put(file_object, 'fabupload.aux')
    remote_sudo(c, f'mv /home/{ssh_user}/fabupload.aux {remote_path}', echo=False, warn=True)


@task(hosts=hosts)
def prepare_deploy(c):
    print_task_header('prepare_deploy')
    local_virtualenv(c, f"python manage.py makemigrations")
    local_virtualenv(c, f"pip freeze > {reqs_path}")
    local_virtualenv(c, f"git add -A", warn=True)
    local_virtualenv(c, f"git commit -m 'fabfile'", warn=True)
    local_virtualenv(c, f"git push", warn=True)


@task(hosts=hosts)
def deploy(c, prepare=True):
    print_task_header('deploy')
    # if not exists(proj_path):
    #     return RuntimeError('Project does not exist, initialize with `fab create`.')

    if prepare:
        prepare_deploy(c)

    remote_virtualenv(c, f"git pull origin main")
    remote_virtualenv(c, f"python3.10 -m pip install -r {proj_path}/{reqs_path}")

    updatetemplates(c)
    remote_python(c, f"manage.py collectstatic -v 0 --noinput")
    remote_python(c, f"manage.py migrate --noinput")

    return restart(c)


# noinspection SqlNoDataSourceInspection,SqlResolve
@task(hosts=hosts)
def create(c, prepare_before_deploying=True):
    print_task_header('create')

    # Generate project locale
    if locale not in remote_shell(c, "locale -a", hide=True).stdout:
        remote_sudo(c, f"locale-gen {locale}")
        remote_sudo(c, f"update-locale {locale}")
        remote_sudo(c, "service postgresql restart_gunicorn")
        remote_shell(c, "exit")

    remote_shell(c, f"mkdir -p {logs_home}")
    remote_shell(c, f"mkdir -p {venv_home}")
    remote_shell(c, f"mkdir -p {proj_path}")

    # Create virtual env
    remote_shell(c, f"cd {venv_home} && python3.10 -m virtualenv {proj_name}")
    remote_virtualenv(c, f"curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10")     # Update pip
    remote_virtualenv(c, f"python3.10 -m pip install gunicorn setproctitle psycopg2-binary django-compressor pymemcache")

    # Create DB and DB user
    db_pwd_sanitized = db_pwd.replace("'", "\'")
    remote_sql(c, f"CREATE USER {proj_name} WITH ENCRYPTED PASSWORD '{db_pwd_sanitized}';")
    remote_sql(c,
               f"CREATE DATABASE {proj_name} WITH OWNER {proj_name} ENCODING = 'UTF8' "
               f"LC_CTYPE = '{locale}' LC_COLLATE = '{locale}' TEMPLATE template0;")

    # Initialize project git with GitHub authentication token
    remote_shell(c, f"mkdir -p {proj_path}")
    remote_virtualenv(c, f"git init")
    remote_virtualenv(c, f"git remote add origin https://github.com/{github_external_repo}")
    remote_virtualenv(c, f"git remote -v")  # Verify

    # Deploy
    deploy(c, prepare=prepare_before_deploying)

    # Start gunicorn service
    remote_sudo(c, f"supervisorctl start gunicorn_{proj_name}")

    # Bootstrap the DB
    addsuperuser(c)

    # Configure rabbitmq vhost and celery
    remote_sudo(c, f"rabbitmqctl add_vhost vhost-{proj_name}", warn=True)
    remote_sudo(c, f"rabbitmqctl add_user '{proj_name}' '{proj_name}'", warn=True)  # Pwd = {proj_name}
    remote_sudo(c, f"rabbitmqctl set_permissions -p 'vhost-{proj_name}' '{proj_name}' '.*' '.*' '.*'", warn=True)

    return True


@task(hosts=hosts)
def remove(c):
    """
    Blow away the current project.
    """
    print_task_header('remove')
    remote_shell(c, f"rm -rf {venv_path}", warn=True)
    remote_shell(c, f"rm -rf {proj_path}", warn=True)
    for template in templates.values():
        if 'remote_path' in template:
            remote_sudo(c, f"rm {template['remote_path']}", warn=True)

    remote_sudo(c, "supervisorctl update")
    remote_sql(c, f"DROP DATABASE IF EXISTS {proj_name};")
    remote_sql(c, f"DROP USER IF EXISTS {proj_name};")


##########
# Utils  #
##########

@task(hosts=hosts)
def stop(c):
    print_task_header('stop')
    """
    Restart gunicorn worker processes for the project.
    """
    remote_shell(c, f"kill -HUP `cat {proj_path}/gunicorn.pid`", warn=True)
    remote_sudo(c, f"supervisorctl stop gunicorn_{proj_name} celerybeat_{proj_name} celeryworker_{proj_name}", warn=True)


@task(hosts=hosts)
def restart(c):
    """
    Restart gunicorn worker processes for the project.
    If the processes are not running, they will be started.
    """
    print_task_header('restart')
    remote_shell(c, f"kill -HUP `cat {proj_path}/gunicorn.pid`", warn=True)
    remote_sudo(c, "supervisorctl reread", warn=True)
    remote_sudo(c, f"supervisorctl restart gunicorn_{proj_name}", warn=True)


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
    sanitized_cmd = cmd.replace("`", r"\`")
    remote_virtualenv(c, f'{sanitized_cmd}')


@task(hosts=hosts)
def logs(c):
    options = ['Quit'] + remote_shell(c, f"ls {logs_home}/", hide=True).stdout.split()
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
                remote_shell(c, f"tail -n 3000 {logs_home}/{options[idx_s]}")
        except BaseException:
            pass


@task(hosts=hosts)
def removelogs(c):
    print_task_header('removelogs')
    remote_shell(c, f"rm -f {logs_home}/*.log*", hide=True)
    restart(c)


@task(hosts=hosts)
def addsuperuser(c):
    print_task_header('addsuperuser')
    if superuser_pwd and superuser_name:
        remote_django(
            c,
            f"from django.contrib.auth import get_user_model;"
            f"User = get_user_model();"
            f"u, _ = User.objects.get_or_create(username='{superuser_name}', email='{superuser_email}');"
            f"u.is_staff = u.is_admin = u.is_superuser = True;"
            f"u.set_password('{superuser_pwd}');"
            f"u.save();", echo=False)


@task(hosts=hosts)
def updatetemplates(c):
    print_task_header('updatetemplates')
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
                        remote_sudo(c, cmd, warn=True)


@task(hosts=hosts)
def replicatedatabase(c):
    dump_fname = f'db_dump_{datetime.now().strftime("%Y%M%d_%H%M%S")}'
    remote_python(c,
        f'manage.py dumpdata --format json --indent 4 --natural-foreign -e contenttypes.ContentType -e auth.Permission --output {dump_fname}.json')
    local_virtualenv(c, f'rsync {hosts[0]}:{proj_path}/{dump_fname}.json {dump_fname}.json')
    local_virtualenv(c, f'rsync -a {hosts[0]}:{proj_path}/static/media/* static/media/')
    remote_virtualenv(c, f'rm {dump_fname}.json')

    local_virtualenv(c, f'python3.10 manage.py flush')
    local_virtualenv(c, f'python3.10 manage.py loaddata {dump_fname}.json')


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

ns = Collection(prepare_deploy, deploy, create, run,
                remove, updatetemplates, restart, stop,
                logs, removelogs, replicatedatabase, addsuperuser)
ns.configure({
    'sudo': {
        'password': ssh_pwd,
    },
    'connect_kwargs': {
        'password': ssh_pwd
    },
})
