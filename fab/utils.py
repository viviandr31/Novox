# coding=utf8
from contextlib import contextmanager
from fabric.api import prefix, cd, sudo, run, lcd, task
from datetime import datetime
import os

@contextmanager
def virtualenv(proj_root, proj):
    with cd(proj_root):
        with prefix('workon %s' % proj):
            yield


def tar1(filename, save_as=None, ignore=None):
    """
    tar and gz dir
    """
    import re
    import os
    import tarfile

    re_check = []
    with open(ignore, 'r') as f:
        for _s in f.readlines():
            _s = _s.strip()
            if _s[0] == '#':
                continue
            _s = _s.replace('.', r'\.')
            _s = _s.replace('*', r'.*')
            re_check.append(re.compile(_s))

    def check(tarinfo):
        _name = os.path.basename(tarinfo.name)
        for _check in re_check:
            if _check.match(_name):
                print 'ignore:', tarinfo.name
                return None
            elif _check.match(tarinfo.name):
                print 'ignore:', tarinfo.name
                return None

        return tarinfo

    print('tar and gz the project ...')
    if not save_as:
        save_as = filename+'.tar.gz'

    cwd = os.path.dirname(filename)
    with lcd(cwd):

        with tarfile.open(save_as, 'w:gz') as tf:
            tf.add(filename, arcname=os.path.basename(filename), filter=check)

    print cwd, save_as, filename
    return save_as


def check_installed(app):
    """check program installed with dpkg, apt-get install"""

    output = run('dpkg -l | grep %s' % app, warn_only=True)
    if not output:
        return False

    print('%s was installed.' % app)
    return True


def check_pip_installed(app, workon=None):
    """check program installed with pip"""
    if workon:
        with prefix('workon %s' % workon):
            output = run('pip list|grep %s' % app, warn_only=True)
            if output:
                return True
        return False

    output = run('pip list|grep %s' % app, warn_only=True)
    if output:
        return True

    return False


def install_mysql(password='techjyt'):
    """install MySQL Server"""
    apps = ['mysql-server-core-5.6',
            'mysql-server-5.6']

    for app in apps:
        if not check_installed(app):
            sudo("echo %s mysql-server/root_password password %s | debconf-set-selections" % (app, password))
            sudo("echo %s mysql-server/root_password_again password %s | debconf-set-selections" % (app, password))
            run("apt-get -y install %s" % app)


def install_apache():
    """install Apache2"""
    app = 'apache2'
    if not check_installed(app):
        sudo("apt-get -y install %s" % app)
        sudo('echo "ServerName localhost">>/etc/apache2/apache2.conf')


def install_mod_wsgi():
    app = 'libapache2-mod-wsgi'
    if not check_installed(app):
        sudo("apt-get -y install libapache2-mod-wsgi")


def install_virtualenv():
    """install virtualenv"""
    apps = [
        'python-pip',
        'python-virtualenv',
        'python-dev',
        'libevent-dev',
        ]
    for app in apps:
        if not check_installed(app):
            sudo('apt-get -y install %s' % app)

    if not check_pip_installed('virtualenvwrapper'):
        sudo('pip install virtualenvwrapper')

        run('echo " ">>.bash_profile')
        run('echo "export WORKON_HOME=~/.virtualenvs">>.bash_profile')
        run('echo "source /usr/local/bin/virtualenvwrapper.sh">>.bash_profile')
        run('source .bash_profile')


def install_mysql_python():
    """install MySQL Python"""
    apps = [
        'libmysqld-dev',
        'libmysqlclient-dev',
        ]

    for app in apps:
        if not check_installed(app):
            sudo('apt-get -y install %s' % app)


def install_libncurses5_dev():
    app = "libncurses5-dev"
    if not check_installed(app):
        sudo('apt-get -y install %s' % app)


def create_virtualenv(proj_name):
    """create virtualenv if it does not exist"""
    if not run('lsvirtualenv|grep %s' % proj_name, warn_only=True):
        run('mkvirtualenv %s' % proj_name)


def init_mysql_db(db_name, db_pw, db_user):
    """initialize MySQL database"""
    run('echo "CREATE DATABASE %s;" | mysql -uroot -p%s'
        % (db_name, db_pw), warn_only=True)
    run('echo "CREATE USER \%s\'@\'localhost\' IDENTIFIED BY \'%s\';" | mysql -uroot -p%s'
        % (db_user, db_pw, db_pw), warn_only=True)
    run('echo "GRANT ALL ON %s.* TO \'%s\'@\'localhost\';" | mysql -uroot -p%s'
        % (db_name, db_user, db_pw), warn_only=True)


def get_time_tag():
    return datetime.today().strftime("%Y%m%d_%H%M")



