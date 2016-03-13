
#---------written by Felix Oesterle (FSO)-----------------
#-DESCRIPTION:
# This is based on fabfile from Raincloud Project (simplified)
#
#-Last modified:  Thu Jul 09, 2015  13:10
#@author Felix Oesterle
#-----------------------------------------------------------
from __future__ import with_statement, print_function
from fabric.api import *
import boto.ec2
from boto.ec2.blockdevicemapping import EBSBlockDeviceType, BlockDeviceMapping
import os
import time
import sys
import socket
import datetime
import math
from collections import defaultdict

#-----------------------------------------------------------
# SHORT DOCU
#-----------------------------------------------------------

# -------- SETUP BOTO and Fabric-----------------
# Virtualenv/Generic pip:
# pip install boto fabric
#
# Conda:
# conda install boto fabric
#
# Debian/Ubuntu:
# apt-get install fabric python-boto
#
# install all other missing modules from list above (start eg. ipython an copy all imports above and see what's missing;
# all modules should be available via pip or easy_install)
#
# Create credentials file: ~/.boto and fill it with the info given by admin (most likely Ben)
# (replace XXXX with what you want to use in fabfile)
#
# [profile XXXX]
# aws_access_key_id = YOUR Access Key ID HERE
# aws_secret_access_key = YOUR Secret Access Key HERE
#
# If you don't want to be prompted to accept ssh keys with every new instance, place these lines into the ~/.ssh/config file:
#
# Host *amazonaws.com
#     User root
#     StrictHostKeyChecking no
#     UserKnownHostsFile /dev/null
#
#
# ------------RUNNING-------------
# look at fabfile.py
#
# to list all possible task of fabfile:
# fab -l
#
# A few first steps:
#     1. go through setup below and adjust at least: ec2Profile, def_logfile
#     2. create instance with
#         fab cloud_make
#     3. Takes between 5 - 10 minutes (if still using spot as def_default_requesttype)
#     4. use
#         fab connect
#        to ssh into instance
#     5. play around with the instance, install software etc
#     6. look at current costs with
#         fab calc_approx_costs_running
#        or list all instances with
#         fab cloud_list
#     7. Once you have enough, shut down your instance via
#         fab terminate_one
#        Or terminate all running instances if you are sure they all belong to you
#         fab cloud_terminate
#        you can also delete volumes with:
#         fab terminate_perm_user_vol:name='your_volume_name'


#-----------------------------------------------------------
# SETUP
#-----------------------------------------------------------
env.disable_known_hosts=True
env.user = 'root'

# FSO--- default name used in tags and instance names:
# set this eg. to your name
def_cn = 'AWS'

# Change to a string identifying yourself
user_identifier = None

# FSO--- ssh and credentials setup
# FSO---the name of the amazon keypair (will be created if it does not exist)
keyn= user_identifier + '_oggm'
# FSO--- the same name as you used in boto setup XXXX (see Readme)
ec2Profile = 'OGGM'
def_key_dir=os.path.expanduser('~/.ssh')

# FSO--- Amazon AWS region setup
def_regions = ['us-east-1','eu-west-1'] #regions for spot search
def_default_avz = 'us-east-1b' #Default availability zone if ondemand is used

# FSO--- type of instance pricing, either:
# ondemand: faster availability, more expensive
# spot: cheaper, takes longer to start up, might be shutdown without warning
def_default_requesttype = 'spot'
# def_default_requesttype = 'ondemand'

# FSO--- the AMI to use
def_ami = dict()
def_ami['eu-west-1'] = 'ami-abc579d8' #eu Ubuntu 14.04 LTS
def_ami['us-east-1'] = 'ami-415f6d2b' #us Ubuntu 14.04 LTS

# Size of the rootfs of created instances
rootfs_size_gb = 50

# Name and size of the persistent /work file system
home_volume_ebs_name = "ebs_" + user_identifier # Set to None to disable home volume
new_homefs_size_gb = 50 # GiB, only applies to newly created volumes

# FSO---log file with timestamps to analyse cloud performance
# look at it with tail -f cloudexecution.log
def_logfile = os.path.expanduser('~/cloudexecution.log')

# Default instance type, index into instance_infos array below
def_inst_type = 1

# Install apt and pip packages for OGGM?
# This can take some time, in particular pip
# After install you should have access to a virtualenv:
# $ workon oggm_env
# in which oggm can run
install_apt = False
install_pip = False

#-----------------------------------------------------------
# SETUP END
#-----------------------------------------------------------
fabfile_dir = os.path.dirname(os.path.abspath(__file__))

if user_identifier is None:
    raise RuntimeError('user identifier must be set')

instance_infos = [
    {
        'type': 't2.micro',
        'vcpus': 1,
        'price': 0.014,
    },
    {
        'type': 'm4.xlarge',
        'vcpus': 4,
        'price': 0.264,
    },
    {
        'type': 'c4.2xlarge',
        'vcpus': 8,
        'price': 0.477,
    },
    {
        'type': 'c4.8xlarge',
        'vcpus': 36,
        'price': 1.906,
    },
]

def_price = instance_infos[def_inst_type]['price']


def update_key_filename(avz):
    key_name = get_keypair_name()
    key_dir = os.path.expanduser(def_key_dir)
    key_dir = os.path.expandvars(key_dir)
    env.key_filename = os.path.join(key_dir, key_name + '.pem')
    print('Current key filename: %s' % env.key_filename)


def find_inst_info(inst_type):
    for info in instance_infos:
        if info['type'] == inst_type:
            return info
    return None


@task
def cloud_make(cn=def_cn):
    """
    Start and prepare instance  -THIS IS THE MAIN ACTIVITY-
    """
    # t = time.time()
    log_with_ts("fabric started ------------------------------")
    log_with_ts("Instance: " + instance_infos[def_inst_type]['type'] + "(" + str(instance_infos[def_inst_type]['vcpus']) + " CPUs)")

    # FSO---set best avz
    if def_default_requesttype == 'spot':
        best_avz,request_type = get_cheapest_availability_zone(def_price)
    else:
        best_avz = def_default_avz
        request_type = 'ondemand'

    print(best_avz, request_type)
    log_with_ts('avz: ' + best_avz)
    log_with_ts('request_type: ' + request_type)

    # FSO--- start instances
    instance_start(cn=cn,avz=best_avz,rt=request_type)
    print("Done setting up instance")
    log_with_ts("instance ready")

    # t_init = time.time()

    # # FSO--- run workflow and get cost of nodes back
    # this is an example, adjust as needed
    # tf = run_workflow(cn=cn,avz=best_avz)

    # # FSO--- get costs of and log
    # costs = calc_approx_costs_running(cn=cn')
    # log_with_ts('Ondemand costs: '+str(costs['ondemand'])+'USD')
    # log_with_ts('Actual costs: '+str(costs['running'])+'USD')

    # # FSO--- terminate instances
    # uncomment if you want to terminate your instances automatically
    # cloud_terminate(cn=cn)
    # log_with_ts("all instances terminated")

    # t_end = time.time()

    # print "Time needed for init (min)", (t_init - t)/60.
    # print "Time needed for workflow and terminate", (t_end - t_init)/60.
    # log_with_ts("fabric end")

@task
def list_ubuntu_amis(regions=def_regions):
    """
    List all available ubuntu 14.04 AMIs in all configured regions
    """
    for region in regions:
        print("Region:", region)
        cloud = boto.ec2.connect_to_region(region,profile_name=ec2Profile)
        imgs = cloud.get_all_images(owners=['099720109477'], filters={'architecture': 'x86_64', 'name': 'ubuntu/images/hvm-ssd/ubuntu-trusty-14.04-amd64-server-*'})
        for img in sorted(imgs, key=lambda v: v.name):
            print(img.id,':',img.name)
        print()


@task
def instance_start(cn=def_cn,
        avz=def_default_avz,
        rt=def_default_requesttype):
    """
    Start and prepare instances
    """
    # FSO---find already existing nodes
    cloud = boto.ec2.connect_to_region(avz[:-1],profile_name=ec2Profile)
    filters = {'tag:type': cn+'node'}
    insta = cloud.get_all_instances(filters=filters)

    # FSO---install each new node
    print("Requesting new instance")
    log_with_ts("Requesting new instance")

    nodenumber = len(insta) + 1

    node_install(cn=cn, avz=avz, rt=rt, idn=nodenumber)

    log_with_ts('Finished installing instance')

    cloud_list()

@task
def run_workflow(cn=def_cn,avz=def_default_avz):
    """
    Runs workflow
    This is just an example, adjust to suit your needs
    """
    # FSO---timing
    t = time.time()
    log_with_ts("workflow start")

    with shell_env( PATH='$PATH:/root/jre1.6.0_45/bin',
            GLOBUS_LOCATION='/root/ws-core-felix',
            ASKALON_HOME='/root/AskalonClient'):
        run('screen -L -S GlobusCont -d -m sh ./GlobusContainerCtrl.sh start ;sleep 1 ')
        run('sh CloudNodes.sh -key '+keyn)

        # FSO--- Clean transfer directory. !HAS! to match the given directory in .agwl file, otherwise useless
        with settings(warn_only=True):
            run('rm -rf ~/public/*')

        # FSO--- check whether Globus container is up
        run('while ! echo exit | nc localhost 40195; do sleep 10; echo "waiting for globus"; done')
        print("Connection to globus")

    log_with_ts("workflow done")
    print("Time needed for run_workflow (min)", (time.time()-t)/60.0)

    #copy result
    # FSO--- TODO need to check decisions in agwl file to get correct files
    target_file = '~/tmp/screenlog.0'
    try:
        get('~/screenlog.0',target_file)
    except:
        print("Uh oh, no screenlog file found")
        log_with_ts("no transfer of results")

    return target_file

def print_instance(inst):
    if inst.state != 'terminated':
        cu_time = datetime.datetime.utcnow()
        it =  datetime.datetime.strptime(inst.launch_time,'%Y-%m-%dT%H:%M:%S.000Z')
    else:
        try:
            cu_time = datetime.datetime.strptime(inst.tags.get('terminate_time'),'%Y-%m-%dT%H:%M:%S.%f')
        except:
            cu_time = datetime.datetime.utcnow()

        it = datetime.datetime.strptime(inst.launch_time,'%Y-%m-%dT%H:%M:%S.000Z')

    time_taken = cu_time - it
    hours, rest = divmod(time_taken.total_seconds(),3600)
    minutes, seconds = divmod(rest, 60)

    print(inst.id, inst.instance_type, \
            inst.tags.get('Name'), \
            inst.tags.get('type'), \
            inst.state, \
            inst.dns_name, \
            inst.private_ip_address, \
            inst.private_dns_name, \
            inst.tags.get('current_price'), \
            inst.tags.get('billable_hours'), \
            inst.tags.get('terminate_time'), \
            inst.placement, \
            'Owner:%s' % inst.tags.get('node-owner'))
    print("running for: ", hours,'h', minutes, "min")


def print_volume(vol):
    info = ""
    if 'vol-lifetime' in vol.tags:
        info += '\tLifetime: ' + vol.tags['vol-lifetime']
    if 'vol-user-name' in vol.tags:
        info += '\tUservolume Name: ' + vol.tags['vol-user-name']
    if 'vol-owner' in vol.tags:
        info += '\tOwner: ' + vol.tags['vol-owner']

    print(vol.id, "\t", vol.zone, "\t", vol.status, '\t', vol.size, info)


@task
def cloud_list(cn=def_cn,itype='all',regions=def_regions):
    """
    List all ec2 instances.
    """
    for region in regions:
        cloud = boto.ec2.connect_to_region(region,profile_name=ec2Profile)
        instances = cloud.get_all_instances()
        vols = cloud.get_all_volumes()
        print()
        print("-------CURRENT RUNNING-----------")
        print("       REGION:", region)
        print()
        print("Instances:")
        print()

        update_costs(cn=cn,regions=regions,itype=itype)

        for reservation in instances:
            for inst in reservation.instances:
                print_instance(inst)
                print()

        print()
        print("Volumes:")
        print()

        for vol in vols:
            print_volume(vol)


def check_keypair(cloud, keynames):
    # Check to see if specified keypair already exists.
    # If we get an InvalidKeyPair.NotFound error back from EC2,
    # it means that it doesn't exist and we need to create it.
    key_dir = def_key_dir
    try:
        cloud.get_all_key_pairs(keynames=[keynames])[0]
    except cloud.ResponseError as e:
        if e.code == 'InvalidKeyPair.NotFound':
            print('Creating keypair: %s' % keynames)
            # Create an SSH key to use when logging into instances.
            key = cloud.create_key_pair(keynames)
            # Make sure the specified key_dir actually exists.
            # If not, create it.
            key_dir = os.path.expanduser(key_dir)
            key_dir = os.path.expandvars(key_dir)
            # if not os.path.isdir(key_dir):
            #     os.mkdir(key_dir, 0700)
            #
            # AWS will store the public key but the private key is
            # generated and returned and needs to be stored locally.
            # The save method will also chmod the file to protect
            # your private key.
            key.save(key_dir)
        else:
            raise


def get_keypair_name():
    key_dir = def_key_dir
    key_dir = os.path.expanduser(key_dir)
    key_dir = os.path.expandvars(key_dir)

    un_file = os.path.join(key_dir, '%s_unique.txt' % keyn)

    if os.path.exists(un_file):
        with open(un_file, 'r') as un:
            unique_part = un.read().strip()
    else:
        import uuid
        unique_part = str(uuid.uuid4().get_hex().upper()[0:8])
        with open(un_file, 'w') as un:
            un.write(unique_part)

    return keyn + '_' + unique_part


def get_user_persist_ebs(cloud, avz):
    if home_volume_ebs_name is None:
        return None

    vols = cloud.get_all_volumes(filters={'tag:vol-user-name':home_volume_ebs_name, 'availability-zone': avz})

    if len(vols) == 0:
        print("Creating new EBS volume for user volume %s" % home_volume_ebs_name)
        vol = cloud.create_volume(new_homefs_size_gb, avz)
        vol.add_tag('vol-user-name', home_volume_ebs_name)
        vol.add_tag('vol-lifetime', 'perm')
        vol.add_tag('vol-owner', user_identifier)
    else:
        vol = vols[0]
        print("Found existing volume %s for user volume %s!" % (vol.id, home_volume_ebs_name))

        if vol.status != 'available':
            print("But it's not available...")
            return None

    return vol


@task
def node_install(cn=def_cn,inst_type_idx=def_inst_type,idn=0,
        avz=def_default_avz,rt=def_default_requesttype,
        group_name='oggmssh',
        ssh_port=22,
        cidr='0.0.0.0/0'):
    """
    Request and prepare single instance
    """
    # FSO---connect
    cloud = boto.ec2.connect_to_region(avz[:-1],profile_name=ec2Profile)
    aminfo = cloud.get_image(def_ami[avz[:-1]])

    # FSO---check if node with same name already exists
    if node_exists(cn + '_node' + str(idn)):
        print("Node already exists")
        sys.exit()

    # Check if ssh keypair exists
    key_name = get_keypair_name()
    check_keypair(cloud, key_name)

    # FSO---create a bigger root device
    dev_sda1 = EBSBlockDeviceType()
    dev_sda1.size = rootfs_size_gb
    dev_sda1.delete_on_termination = True
    bdm = BlockDeviceMapping()
    bdm['/dev/sda1'] = dev_sda1

    dev_sdf_vol = get_user_persist_ebs(cloud, avz)

    # Check to see if specified security group already exists.
    # If we get an InvalidGroup.NotFound error back from EC2,
    # it means that it doesn't exist and we need to create it.
    try:
        group = cloud.get_all_security_groups(groupnames=[group_name])[0]
    except cloud.ResponseError as e:
        if e.code == 'InvalidGroup.NotFound':
            print('Creating Security Group: %s' % group_name)
            # Create a security group to control access to instance via SSH.
            group = cloud.create_security_group(group_name, 'A group that allows SSH access')
        else:
            raise

    # Add a rule to the security group to authorize SSH traffic
    # on the specified port.
    try:
        group.authorize('tcp', ssh_port, ssh_port, cidr)
    except cloud.ResponseError as e:
        if e.code == 'InvalidPermission.Duplicate':
            print('Security Group: %s already authorized' % group_name)
        else:
            raise

    log_with_ts("request node "+str(idn))
    print('Reserving instance for node', aminfo.id, instance_infos[inst_type_idx]['type'], aminfo.name, aminfo.region)

    if rt == 'spot':
        print("placing node in ",avz)
        requests = cloud.request_spot_instances(def_price,
                      def_ami[avz[:-1]],
                      count=1,
                      type='one-time',
                      security_groups=[group_name],
                      key_name=key_name,
                      placement=avz,
                      instance_type=instance_infos[inst_type_idx]['type'],
                      block_device_map=bdm)
        req_ids = [request.id for request in requests]
        instance_ids = wait_for_fulfillment(cloud,req_ids)
        instances = cloud.get_only_instances(instance_ids=instance_ids)
        node = instances[0]
        log_with_ts("fullfilled spot node "+str(idn))
    else:
        print("placing node in ",avz)
        reservation = cloud.run_instances(image_id=def_ami[avz[:-1]],
                key_name=key_name,
                placement = avz,
                security_groups=[group_name],
                instance_type=instance_infos[inst_type_idx]['type'],
                block_device_map= bdm)
        node = reservation.instances[0]
        log_with_ts("fullfilled ondemand node "+str(idn))

    time.sleep(2)
    while not node.update() == 'running':
        print('waiting for', cn, 'node', idn, 'to boot...')
        time.sleep(5)

    log_with_ts("booted node "+str(idn))

    if dev_sdf_vol is not None:
        cloud.attach_volume(dev_sdf_vol.id, node.id, "/dev/sdf")

    node.add_tag('Name', cn+'_node'+str(idn))
    node.add_tag('type', cn+'node')
    node.add_tag('node-owner', user_identifier)

    # FSO---set delete on termination flag to true for ebs block device
    node.modify_attribute('blockDeviceMapping', { '/dev/sda1' : True })

    # FSO--- test socket connect to ssh service
    ssh_test(node)
    log_with_ts("reachable node "+str(idn))

    update_key_filename(node.region.name)

    # FSO---enable root (also sets env.host_name needed for fab to run run())
    enable_root(node.dns_name)

    # Mount potential user volume
    if dev_sdf_vol is not None:
        use_user_volume(node.dns_name)

    env.user = 'root'

    if install_apt:
        # This installs the apt packages necessary for OGGM py3
        with hide('output'):
            run('apt-get -u update; sleep 4', shell=True, pty=True)
            run('apt-get install -y build-essential liblapack-dev gfortran libproj-dev', shell=True, pty=True)
            run('apt-get install -y gdal-bin libgdal-dev', shell=True, pty=True)
            run('apt-get install -y netcdf-bin ncview python-netcdf', shell=True, pty=True)
            run('apt-get install -y tk-dev python3-tk python3-dev ttf-bitstream-vera', shell=True, pty=True)

            run('apt-get install -y python-pip', shell=True, pty=True)
            run('apt-get install -y git', shell=True, pty=True)

            # AWS stuff, only if installer has AWS cli installed
            aws_file = os.path.expanduser('~/.aws/config')
            if os.path.exists(aws_file):
                run('apt-get install -y awscli', shell=True, pty=True)
                run('mkdir ~/.aws')
                put(aws_file, '~/.aws/config')

            # Now virtualenv stuffs
            run('pip install virtualenvwrapper')
            run('mkdir ~/.pyvirtualenvs')

            run("echo '' >> ~/.profile")
            run("echo '# Virtual environment options' >> ~/.profile")
            run("echo 'export WORKON_HOME=$HOME/.pyvirtualenvs' >> ~/.profile")
            run("echo 'source /usr/local/bin/virtualenvwrapper_lazy.sh' >> ~/.profile")

            run("echo '' >> ~/.bashrc")
            run("echo '# Virtual environment options' >> ~/.bashrc")
            run("echo 'export WORKON_HOME=$HOME/.pyvirtualenvs' >> ~/.bashrc")
            run("echo 'source /usr/local/bin/virtualenvwrapper_lazy.sh' >> ~/.bashrc")

            run('mkvirtualenv oggm_env -p /usr/bin/python3')

    # Python install script
    pip_file = os.path.join(fabfile_dir, 'install_env')
    if os.path.exists(pip_file):
        put(pip_file, '~/install_env')
        run('chmod a+x ~/install_env')

    if install_pip:
        run('./install_env')
        # After pip install one should replace mpl backend
        fpath = '/root/.pyvirtualenvs/oggm_env/lib/python3.4/site-packages/matplotlib/mpl-data/matplotlibrc'
        run("sed -i 's/^backend.*/backend      : Agg/' " + fpath)

    log_with_ts("finished node "+str(idn))


@task
def terminate_perm_user_vol(name=home_volume_ebs_name,regions=def_regions):
    """
    Terminate the permanent user volume
    """
    print(regions)
    for region in regions:
        cloud = boto.ec2.connect_to_region(region, profile_name=ec2Profile)
        vols = cloud.get_all_volumes(filters={'tag:vol-user-name':name})
        for vol in vols:
            if vol.status == 'available':
                print(vol.id,"\t", vol.status, "... deleted")
                vol.delete()
            else:
                print(vol.id,"\t", vol.status, "... in use")


@task
def cloud_terminate(cn=def_cn,itype='all',regions=def_regions):
    """
    Terminate all instances
    """
    print(regions)
    for region in regions:
        print()
        print("-------CURRENT RUNNING-----------")
        print("       REGION:",region)

        cloud = boto.ec2.connect_to_region(region, profile_name=ec2Profile)
        instances = cloud.get_all_instances()
        vol = cloud.get_all_volumes()

        update_costs(cn=cn,itype=itype)

        for reservation in instances:
            for inst in reservation.instances:
                if inst.state != 'terminated':
                    if itype == 'all':
                        print('TERMINATING', inst.tags.get('Name'), inst.dns_name)
                        inst.add_tag('Name', 'term')
                        inst.add_tag('type', 'term')
                        inst.terminate()
                        stati2 = datetime.datetime.utcnow()
                        inst.add_tag('terminate_time', stati2.isoformat())
                    elif itype == 'node' and inst.tags.get('type') == cn+'node':
                        print('TERMINATING', inst.tags.get('Name'), inst.dns_name)
                        inst.add_tag('Name', 'term')
                        inst.add_tag('type', 'term')
                        inst.terminate()
                        stati2 = datetime.datetime.utcnow()
                        inst.add_tag('terminate_time', stati2.isoformat())
                    elif itype == 'master' and inst.tags.get('type') == cn+'master':
                        print('TERMINATING', inst.tags.get('Name'), inst.dns_name)
                        inst.add_tag('Name', 'term')
                        inst.add_tag('type', 'term')
                        inst.terminate()
                        stati2 = datetime.datetime.utcnow()
                        inst.add_tag('terminate_time', stati2.isoformat())

        for unattachedvol in vol:
            if 'vol-lifetime' in unattachedvol.tags and unattachedvol.tags['vol-lifetime'] == 'perm':
                print(unattachedvol.id,"\t", unattachedvol.status, "... is marked permanent")
            elif unattachedvol.status == 'available':
                print(unattachedvol.id,"\t", unattachedvol.status, "... deleted")
                unattachedvol.delete()
            else:
                print(unattachedvol.id,"\t", unattachedvol.status, "... not deleted")


@task
def terminate_one(regions=def_regions, nn=''):
    """
    Terminate one instance
    """
    instlist = list()
    i = 0
    for region in regions:
        print()
        print("-------CURRENT RUNNING-----------")
        print("       REGION: ", region)
        print()

        cloud = boto.ec2.connect_to_region(region, profile_name=ec2Profile)
        reservations = cloud.get_all_instances()

        for reserv in reservations:
            for inst in reserv.instances:
                if inst.state == 'terminated':
                    continue

                print('Instance %s:' % i)
                print_instance(inst)
                print()

                instlist.append(inst)
                i += 1

    print()

    if nn == '':
        nn = prompt('Which instance to terminate:')

    nn = int(nn)

    inst = instlist[nn]
    inst.add_tag('Name', 'term')
    inst.add_tag('type', 'term')
    inst.terminate()
    stati2 = datetime.datetime.utcnow()
    inst.add_tag('terminate_time', stati2.isoformat())


@task
def terminate_volume(regions=def_regions, nn=''):
    """
    Terminate one volume
    """
    vollist = list()
    i = 0
    for region in regions:
        print()
        print("-------CURRENT RUNNING-----------")
        print("       REGION: ", region)
        print()

        cloud = boto.ec2.connect_to_region(region, profile_name=ec2Profile)
        vols = cloud.get_all_volumes()

        for vol in vols:
            print("Volume %s:" % i)
            print_volume(vol)
            print()

            vollist.append(vol)
            i += 1

    print()

    if nn == '':
        nn = prompt('Which volume to terminate:')

    nn = int(nn)

    vollist[nn].delete()


@task
def calc_approx_costs_running(cn=def_cn,regions=def_regions,itype ='all'):
    """
    calculate compute costs (network or storage not included)
    only running instances are considered
    From amazon: The instances will be billed at the then-current Spot Price regardless of the actual bid
    """

    # FSO---update the price tags for each node
    update_costs(cn=cn,regions=regions,itype=itype)

    costs = dict()
    costs['running'] = 0.0
    costs['ondemand'] = 0.0

    for region in regions:
        cloud = boto.ec2.connect_to_region(region,profile_name=ec2Profile)
        instances = cloud.get_all_instances()
        print()
        print("----------REGION:",region,itype,'-----------')


        for reservation in instances:
            for inst in reservation.instances:
                if inst.state == 'running' and (inst.tags.get('type')==cn+itype or itype=='all'):
                    hours = float(inst.tags.get('billable_hours'))
                    cu_price = float(inst.tags.get('current_price'))
                    cu_ondemand_price = hours * find_inst_info(inst.instance_type)['price']

                    print()
                    print(inst.id, inst.instance_type, \
                                inst.tags.get('Name'), \
                                inst.dns_name,\
                                inst.tags.get('current_price')+'USD', \
                                inst.tags.get('billable_hours')+'h', \
                                inst.placement)
                    # print 'Billable hours ',hours
                    # print 'Current price', cu_price
                    # print 'Current ondemand price', cu_ondemand_price
                    costs['ondemand'] += cu_ondemand_price
                    if inst.spot_instance_request_id is None:
                        print('ondemand instance')
                        costs['running'] = cu_ondemand_price
                    else:
                        print('spot instance')
                        costs['running'] += cu_price

    print()
    print('Total ondemand: ', costs['ondemand'])
    print('Total of running: ' , costs['running'])

    return costs

@task
def connect(nn='', user='root'):
    """
    SSH to cloud instances (as root)
    """
    instlist = list()
    i = 0
    for region in def_regions:
        cloud = boto.ec2.connect_to_region(region,profile_name=ec2Profile)
        instances = cloud.get_all_instances()
        print()
        for reservation in instances:
            for inst in reservation.instances:
                if inst.state == 'running':
                    print(i, ': ', inst.tags.get('Name'),inst.dns_name, inst.private_ip_address, inst.placement)
                    instlist.append(inst)
                    i += 1

    print()

    if nn =='':
        nn = prompt('Which instance to connect to:')

    update_key_filename(instlist[int(nn)].region.name)

    print('ssh', '-i', os.path.expanduser(env.key_filename), '%s@%s' % (user,instlist[int(nn)].dns_name))
    print('...')
    print()

    os.execlp('ssh', 'ssh', '-i', os.path.expanduser(env.key_filename), '%s@%s' % (user, instlist[int(nn)].dns_name))


def get_cheapest_availability_zone(ondemand_price):
    """
    get the cheapest avz and check if below ondemand_price
    BEWARE: does not necessarily get the cheapest avz at the moment, but the one with the lowest maxium price
    in the last 24 hours. Hopefully that's the most stable price-wise
    """
    avz_prices_nodes = defaultdict(list)
    for region in def_regions:
        cloud = boto.ec2.connect_to_region(region,profile_name=ec2Profile)
        stati2 = datetime.datetime.utcnow()
        stati1 = stati2 - datetime.timedelta(hours=3)
        prices = cloud.get_spot_price_history(
            # instance_type=def_itype,
            # instance_type=['m1.small','m1.medium'],
            start_time=stati1.isoformat(),
            end_time= stati2.isoformat(),
            product_description='Linux/UNIX')

        # FSO---split in availability_zones
        for price in prices:
            if price.instance_type == instance_infos[def_inst_type]['type']:
                avz_prices_nodes[str(price.availability_zone)].append(price)

    # FSO---remove us-east-1c as access is constrained
    try:
        del avz_prices_nodes['us-east-1c']
    except:
        print( "no us-east-1c")

    maxprice_nodes = dict()
    for key in avz_prices_nodes:
        allpr_nodes = [k.price for k in avz_prices_nodes[key]]
        maxprice_nodes[key] = max(allpr_nodes)

    best_avz_nodes = min(maxprice_nodes, key=maxprice_nodes.get) # gets just the first if serveral avz's are the same

    print("Cheapest nodes: ", best_avz_nodes, maxprice_nodes[best_avz_nodes])
    print("Ondemand nodes (EU):", ondemand_price)
    if maxprice_nodes[best_avz_nodes] < ondemand_price:
        return best_avz_nodes,'spot'
    else:
        return def_default_avz,'ondemand'

def wait_for_fulfillment(cloud,pending_ids):
    """
    Wait for fulfillment of spot instance requests
    """
    instances = list()
    while not len(pending_ids) == 0:
        pending = cloud.get_all_spot_instance_requests(pending_ids)
        for request in pending:
            if request.status.code == 'fulfilled':
                pending_ids.pop(pending_ids.index(request.id))
                print("spot request `{}` fulfilled!".format(request.id))
                #print request.__dict__
                instances.append(request.instance_id)
                cloud.cancel_spot_instance_requests(request.id)
            elif request.state == 'cancelled':
                pending_ids.pop(pending_ids.index(request.id))
                print("spot request `{}` cancelled!".format(request.id))
            else:
                print("waiting on `{}`".format(request.id))
        time.sleep(5)
    print("all spots fulfilled!")
    return instances

def update_costs(cn=def_cn,itype='all',regions=def_regions):
    """
    Updates the price tags of all running instances
    """
    for region in regions:
        cloud = boto.ec2.connect_to_region(region,profile_name=ec2Profile)
        instances = cloud.get_all_instances()

        for reservation in instances:
            for inst in reservation.instances:
                total_price = 0.0
                if inst.state != 'terminated':
                    cu_time = datetime.datetime.utcnow()
                    it =  datetime.datetime.strptime(inst.launch_time,'%Y-%m-%dT%H:%M:%S.000Z')
                    time_taken = cu_time - it
                    hours = int(math.ceil(time_taken.total_seconds()/3600.))

                    # FSO---for spot instances
                    if inst.spot_instance_request_id is not None:
                        # FSO---loop through hours. spot instances are billed according to the price at each full hour!
                        for i in range(hours):
                            price = cloud.get_spot_price_history(instance_type=inst.instance_type,
                                    start_time = it.isoformat(),
                                    end_time= it.isoformat(),
                                    product_description='Linux/UNIX',
                                    availability_zone=inst.placement)
                            # print "Hour: ", it, "price=",price
                            it = it + datetime.timedelta(hours=1)
                            total_price = total_price + price[0].price
                    # FSO---ondemand instances
                    else:
                        total_price = hours * find_inst_info(inst.instance_type)['price']

                    inst.add_tag('current_price', total_price)
                    inst.add_tag('billable_hours', hours)

def log_with_ts(logtext="no text given",lf=def_logfile):
    """
    Helper function to write logs with timestamps
    """
#	logtime = time.time()
#   st = datetime.datetime.fromtimestamp(logtime).strftime('%Y-%m-%d %H:%M:%S')
    st = str(datetime.datetime.utcnow())
    with open(lf, "a+") as myfile:
        myfile.writelines('['+st+' UTC] '+ logtext+'\n')

def spot_price(cloud,launch_time,inst_type):
    """
    Helper function to get spot price"
    """
    prices = dict()
    #stati = datetime.datetime.utcnow()
    #stati = stati - datetime.timedelta(hours=1)
    #print stati
    # Get prices for instance, AZ and time range
    price = cloud.get_spot_price_history(instance_type=inst_type,
#           start_time=stati.isoformat(),
            start_time = launch_time,
            end_time= launch_time,
            product_description='Linux/UNIX',
            availability_zone='eu-west-1a')
    prices['a'] = price[0].price
    price = cloud.get_spot_price_history(instance_type=inst_type,
            start_time = launch_time,
            end_time= launch_time,
            product_description='Linux/UNIX',
            availability_zone='eu-west-1b')
    prices['b'] = price[0].price
    price = cloud.get_spot_price_history(instance_type=inst_type,
            start_time = launch_time,
            end_time= launch_time,
            product_description='Linux/UNIX',
            availability_zone='eu-west-1c')
    prices['c'] = price[0].price

    cloudus = boto.ec2.connect_to_region('us-east-1')
    price = cloudus.get_spot_price_history(instance_type=inst_type,
            start_time = launch_time,
            end_time= launch_time,
            product_description='Linux/UNIX',
            availability_zone='us-east-1c')
    if len(price) > 0:
        prices['usc'] = price[0].price
    else:
        prices['usc'] = 0.0

    price = cloudus.get_spot_price_history(instance_type=inst_type,
            start_time = launch_time,
            end_time= launch_time,
            product_description='Linux/UNIX',
            availability_zone='us-east-1b')
    if len(price) > 0:
        prices['usb'] = price[0].price
    else:
        prices['usb'] = 0.0
    #for price in price:
        #print price.timestamp, price.price
    return prices

def node_find(node,avz=def_default_avz):
    """
    Return the instance object of a given node hostname.
    """
    cloud = boto.ec2.connect_to_region(avz[:-1])
    instances = cloud.get_all_instances()
    for reservation in instances:
        for inst in reservation.instances:
            if inst.tags.get('Name') == node and inst.state == 'running':
                print('found', inst.tags.get('Name'), inst.dns_name)
                return inst

def node_exists(node,avz=def_default_avz):
    """
    checks if node with given name exists
    """
    cloud = boto.ec2.connect_to_region(avz[:-1],profile_name=ec2Profile)
    instances = cloud.get_all_instances()
    for reservation in instances:
        for inst in reservation.instances:
            if inst.tags.get('Name') == node and inst.state == 'running':
                print('found', inst.tags.get('Name'), inst.dns_name)
                return True

    return False

def enable_root(host):
    """
    Enable root access on instance
    """
    env.host_string = host
    env.user = 'ubuntu'
    run("sudo perl -i -pe 's/disable_root: 1/disable_root: 0/' /etc/cloud/cloud.cfg")
    run("sudo perl -i -pe 's/#PermitRootLogin .*/PermitRootLogin without-password/' /etc/ssh/sshd_config")
    run('sudo cp -f /home/ubuntu/.ssh/authorized_keys /root/.ssh/authorized_keys', shell=True, pty=True)
    run("sudo reload ssh")

def use_user_volume(host):
    """
    Setup and mount user /work volume
    """
    env.host_string = host
    env.user = 'ubuntu'
    run("test -e /dev/xvdf1 || ( sudo sgdisk -o -g -n 1:2048:0 /dev/xvdf && sudo mkfs.ext4 /dev/xvdf1 )")
    run("sudo mkdir /work")
    run("sudo mount /dev/xvdf1 /work")
    run("test -e /work/ubuntu || ( sudo mkdir /work/ubuntu && sudo chown ubuntu:ubuntu /work/ubuntu )")


def ssh_test(inst):
    """
    checks for ssh connectability
    """
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(4)
            sock.connect((inst.dns_name, 22))
            break
        except:
            print('waiting for ssh daemon...')
            time.sleep(5)
        finally:
            sock.close()



