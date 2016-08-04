# coding=utf-8
from fabric.api import task, sudo, put
from utils import check_installed
import os


@task
def start():
    """start apache2 service"""
    sudo('service apache2 start')


@task
def reload():
    """reload apache2 service"""
    sudo('service apache2 reload')

@task
def stop():
    """stop apache2 service"""
    sudo('service apache2 stop')

@task
def restart():
    """restart apache2 service"""
    sudo('service apache2 restart')

@task
def setup():
    """install Apache2 and its mod_wsgi module"""
    apps = ['apache2', 'libapache2-mod-wsgi']
    for app in apps:
        if not check_installed(app):
            sudo("apt-get -y install %s" % app)

# /%(proj_name)s
# WSGIPythonPath %(proj_root)s:/home/%(user)s/.virtualenvs/%(proj_name)s/lib/python2.7/site-packages

TEMPLATE_CONFIG = """
WSGIDaemonProcess %(proj_name)s python-path=%(proj_root)s:/home/%(user)s/.virtualenvs/%(proj_name)s/lib/python2.7/site-packages
WSGIProcessGroup %(proj_name)s
<VirtualHost *:%(site_port)s>
  WSGIScriptAlias / %(proj_root)s/%(proj_name)s/wsgi.py

  ErrorLog /var/www/%(www_src)s/log/apache2/error.log
  CustomLog /var/www/%(www_src)s/log/apache2/access.log combined

  <Directory %(proj_root)s/%(proj_name)s>
     <Files wsgi.py>
       Require all granted
     </Files>
  </Directory>

  Alias /static /var/www/%(www_src)s/static
  Alias /media /var/www/%(www_src)s/media

  <Directory /var/www/%(www_src)s/static>
    #Order deny,allow
    #Allow from all
    Options FollowSymLinks
    Require all granted
  </Directory>

  <Directory /var/www/%(www_src)s/media>
    #Options FollowSymLinks
    Require all granted
  </Directory>
</VirtualHost>
"""


def get_config(env):
    """setup Apache2 conf files"""
    c = TEMPLATE_CONFIG % env
    return c

