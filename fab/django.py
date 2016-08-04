# coding=utf-8
from fabric.api import task, sudo, put, env, run
from fabric.contrib.files import exists, upload_template
import os

@task
def setup_site():
    """
      prepare Django project site conf file
        prerequisite:
            env.user
            env.proj_root
            env.proj_name
        operations:
            confs/apache2.conf % env
            --->
            /etc/apache2/sites-avilable/%(proj_name)s.conf % env
    """
    upload_template('./confs/apache2.conf', '/etc/apache2/sites-available/%(proj_name)s.conf' % env, context=env, use_sudo=True)
    enabled_sites = run('ls /etc/apache2/sites-enabled')

    # 这样会关闭其他端口运行的网站
    #print 'Current enabled sites: %s\n' % enabled_sites
    #sudo('a2dissite %s' % enabled_sites)

    sudo('a2ensite %(proj_name)s.conf' % env)


@task
def setup():
    """
        setup Apache2 site and folder structures for Django
        1. Create and enable <project_name>.confs
        2. Link /var/www/<project_name>/media
        3. Create /var/www/<project_name>/static
        4. Create virtual env
    """
    setup_site()
    try:
        if not exists('/var/www/%(proj_name)s' % env):
            sudo('mkdir /var/www/%(proj_name)s' % env)
        if not exists('/var/www/%(proj_name)s/media' % env):
            sudo('ln -sfn %(proj_root)s/media /var/www/%(proj_name)s/media'
                 % env)
        if not exists('/var/www/%(proj_name)s/static' % env):
            sudo('mkdir /var/www/%(proj_name)s/static' % env)
        sudo('chown -R %(user)s /var/www/%(proj_name)s' % env)
        if not exists('%(proj_root)s' % env):
            run('mkdir -p %(proj_root)s' % env)
    except KeyError:
        print 'Env not defined'
