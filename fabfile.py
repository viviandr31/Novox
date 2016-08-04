# coding=utf8
from fabric.api import *
from fab import apache as ap
from fab import django as dj
from fab import utils as ut
import os
from fab.utils import *
from fabric.contrib.files import exists, sed
from fabric.contrib.project import rsync_project


env.user = 'ubuntu'
env.domain = '54.200.101.157'
env.hosts = ['54.200.101.157']

env.proj_name = os.path.basename(os.path.dirname(__file__))
env.proj_root = '/home/' + env.user + '/projects/'+ env.proj_name
env.key_filename = ['./hanalab.pem']

ignore = '.fabfileignore'
proj_path = '~/projects'


@task
def pull_log():
    """pull log from the server"""
    get('/var/log/django/error.log', 'logs/dj.error.log')
    get('/var/log/apache2/error.log', 'logs/ap.error.log')
    get('/var/log/apache2/access.log', 'logs/ap.access.log')


def rsync_repo():
    """rsync project dirctory to server"""
    rsync_project(local_dir='.', remote_dir=env.proj_root, extra_opts='--exclude-from=%s' % ignore)


def backup_repo():
    """backup server existing project directory"""
    with cd(proj_path):
        run('tar -zcf %s_%s.tar.gz %s' % (env.proj_name, get_time_tag(), env.proj_name),
            warn_only=True)


def dev2prod():
    sed('%(proj_root)s/%(proj_name)s/settings.py' % env,
        before='DEBUG = True',
        after='DEBUG = False',
        backup='')


@task
def deploy():
    """deploy Django project"""
    pre_deploy()
    rsync_repo()
    post_deploy()


def pre_deploy():
    # run('export PRODUCTION=1')
    ap.stop()
    backup_repo()


def post_deploy():
    dev2prod()
    with virtualenv(env.proj_root, env.proj_name):
        run("pip install -r requirements.txt")
        sudo('python manage.py collectstatic --noinput')
        sudo('chmod -R 777 db')
        sudo('chmod -R 777 media')
        # sudo('chown -R %(user)s /var/log/django' % env)
        # run('nohup python bin/device-alive-report.py &')
    ap.start()
    print 'All Done! You may visit %(hosts)s now' % env


@task
def db_pull():
    """download db.sqlite3"""
    with cd(env.proj_root):
        get('db/db.sqlite3', 'db/host_db.sqlite3')


@task
def init_env():
    """init project site"""
    dj.setup()

