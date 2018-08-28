
#---------written by Felix Oesterle (FSO)-----------------
#-DESCRIPTION:
# This is based on fabfile from Raincloud Project (simplified)
#
#-Last modified:  Thu Jul 09, 2015  13:10
#@author Felix Oesterle
#-----------------------------------------------------------
# pragma: no cover
# flake8: noqa
from __future__ import with_statement, print_function
from fabric.api import *
import boto.ec2
from boto.vpc import VPCConnection
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
#     1. Go through setup below and adjust at least: ec2Profile, def_logfile
#     2. Create instance with
#         fab cloud_make
#        If you are using spot instances and require your instances to be in the same region
#         fab instance_start
#        This will use the region configured in def_default_avz.
#     3. Takes between 5 - 10 minutes (if still using spot as def_default_requesttype)
#     4. Use
#         fab install_node_software
#        to setup a virtualenv ready for OGGM.
#
#        If you already setup a virtualenv on your user volume
#         fab install_node_apt
#        to install only required system components.
#     5. Use
#         fab connect
#        to ssh into instance
#     6. play around with the instance, install software etc
#     7. look at current costs with
#         fab calc_approx_costs_running
#        or list all instances with
#         fab cloud_list
#     8. Once you have enough, shut down your instance via
#         fab terminate_one
#        Or terminate all running instances if you are sure they all belong to you
#         fab cloud_terminate
#        you can also delete volumes with:
#         fab terminate_perm_user_vol:name='your_volume_name'


#-----------------------------------------------------------
# SETUP
#-----------------------------------------------------------
env.disable_known_hosts=True
env.user = 'ubuntu'

# FSO--- default name used in tags and instance names:
# set this eg. to your name
def_cn = 'AWS'

# Change to a string identifying yourself
user_identifier = None

# FSO--- ssh and credentials setup
# FSO---the name of the amazon keypair (will be created if it does not exist)
keyn=(user_identifier or 'None') + '_oggm'
# FSO--- the same name as you used in boto setup XXXX (see Readme)
ec2Profile = 'OGGM'
def_key_dir=os.path.expanduser('~/.ssh')

# FSO--- Amazon AWS region setup
def_regions = ['us-east-1','eu-west-1'] #regions for spot search
def_default_avz = 'eu-west-1a' #Default availability zone if ondemand is used

# FSO--- type of instance pricing, either:
# ondemand: faster availability, more expensive
# spot: cheaper, takes longer to start up, might be shutdown without warning
def_default_requesttype = 'spot'
# def_default_requesttype = 'ondemand'

# FSO--- the AMI to use
def_ami = dict()
def_ami['eu-west-1'] = 'ami-c32610a5' #eu Ubuntu 16.04 LTS oggm-base
def_ami['us-east-1'] = 'ami-9f3e9689' #us Ubuntu 16.04 LTS oggm-base

# Subnet to use per AVZ, expects a tuple (vpc-id, subnet-id)
def_subnet = dict()
def_subnet['eu-west-1a'] = ('vpc-61f04204', 'subnet-306ff847')
def_subnet['eu-west-1b'] = ('vpc-61f04204', 'subnet-6ad17933')
def_subnet['us-west-1c'] = ('vpc-61f04204', 'subnet-2e2f414b')

# Size of the rootfs of created instances
rootfs_size_gb = 50

# Name and size of the persistent /work file system
home_volume_ebs_name = "ebs_" + (user_identifier or 'None') # Set to None to disable home volume
new_homefs_size_gb = 50 # GiB, only applies to newly created volumes

# FSO---log file with timestamps to analyse cloud performance
# look at it with tail -f cloudexecution.log
def_logfile = os.path.expanduser('~/cloudexecution.log')

# Default instance type, index into instance_infos array below
def_inst_type = 1


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


def update_key_filename(region):
    key_name = get_keypair_name(region)
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
        imgs = cloud.get_all_images(owners=['099720109477'], filters={'architecture': 'x86_64', 'name': 'ubuntu/images/hvm-ssd/ubuntu-xenial-16.04-amd64-server-*'})
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
            'Subnet:%s' % inst.subnet_id, \
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


def get_keypair_name(region):
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

    return keyn + '_' + region + '_' + unique_part


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
    vpcconn = VPCConnection(region=cloud.region, profile_name=ec2Profile)

    try:
        vpc_id, subnet_id = def_subnet[avz]
        vpc = vpcconn.get_all_vpcs(vpc_ids=[vpc_id])[0]
    except:
        vpc_id = None
        subnet_id = None
        vpc = None

    # FSO---check if node with same name already exists
    if node_exists(cn + '_node' + str(idn)):
        print("Node already exists")
        sys.exit()

    # Check if ssh keypair exists
    key_name = get_keypair_name(avz[:-1])
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

    # Authorize all Intra-VPC traffic
    if vpc is not None:
        try:
            group.authorize('-1', -1, -1, vpc.cidr_block)
        except cloud.ResponseError as e:
            if e.code != 'InvalidPermission.Duplicate':
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
                      security_group_ids=[group.id],
                      key_name=key_name,
                      placement=avz,
                      subnet_id=subnet_id,
                      ebs_optimized=True,
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
                placement=avz,
                subnet_id=subnet_id,
                security_group_ids=[group.id],
                ebs_optimized=True,
                instance_type=instance_infos[inst_type_idx]['type'],
                block_device_map=bdm)
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

    # Mount potential user volume
    if dev_sdf_vol is not None:
        use_user_volume(node.dns_name)

    log_with_ts("finished node "+str(idn))


@task
def install_node_software(nn=''):
    """
    Setup ready-for-use virtualenv for OGGM on instance
    """
    inst = select_instance(nn)
    install_node_apt('', inst)
    install_node_pip('', inst)

    run('echo Rebooting... && sleep 1 && sudo shutdown -r now')


@task
def install_node_pip(nn='', inst=None):
    """
    Install oggm dependencies via pip
    """
    if inst is None:
        inst = select_instance(nn)
    update_key_filename(inst.region.name)
    env.host_string = inst.dns_name
    env.user = 'ubuntu'

    run("""
    export LC_ALL=C &&
    source ~/.virtenvrc &&
    workon oggm_env &&
    pip install --upgrade pip &&
    pip install numpy &&
    pip install scipy &&
    pip install pandas shapely cython &&
    pip install matplotlib &&
    pip install gdal==1.11.2 --install-option="build_ext" --install-option="--include-dirs=/usr/include/gdal" &&
    pip install fiona --install-option="build_ext" --install-option="--include-dirs=/usr/include/gdal" &&
    pip install mpi4py &&
    pip install pyproj rasterio Pillow geopandas netcdf4 scikit-image configobj joblib xarray boto3 motionless pytest progressbar2 &&
    pip install git+https://github.com/fmaussion/salem.git &&
    sed -i 's/^backend.*/backend      : Agg/' "${WORKON_HOME}"/oggm_env/lib/python?.?/site-packages/matplotlib/mpl-data/matplotlibrc
    """, pty=False)


@task
def install_node_apt(nn='', inst=None):
    """
    Install required OGGM apt dependencies
    """
    if inst is None:
        inst = select_instance(nn)
    update_key_filename(inst.region.name)
    env.host_string = inst.dns_name
    env.user = 'ubuntu'

    run("""
    export LC_ALL=C &&
    export DEBIAN_FRONTEND=noninteractive &&
    sudo apt-get -y update &&
    sudo apt-get -y dist-upgrade &&
    sudo apt-get -y install build-essential liblapack-dev gfortran libproj-dev gdal-bin libgdal-dev netcdf-bin ncview python3-netcdf4 tk-dev python3-tk python3-dev python3-numpy-dev ttf-bitstream-vera python3-pip git awscli virtualenvwrapper openmpi-bin libopenmpi-dev
    """, pty=False)

    copy_files = ['~/.aws/credentials', '~/.aws/config', '~/.screenrc', '~/.gitconfig']

    for cf in copy_files:
        if not os.path.exists(os.path.expanduser(cf)):
            continue
        run('mkdir -p %s' % os.path.dirname(cf))
        put(cf, cf)

    run("""
    if [ -e /work/ubuntu ]; then
        mkdir -p /work/ubuntu/.pyvirtualenvs
        echo '# Virtual environment options' > ~/.virtenvrc
        echo 'export WORKON_HOME="/work/ubuntu/.pyvirtualenvs"' >> ~/.virtenvrc
        echo 'source /usr/share/virtualenvwrapper/virtualenvwrapper_lazy.sh' >> ~/.virtenvrc
    else
        mkdir -p ~/.pyvirtualenvs
        echo '# Virtual environment options' > ~/.virtenvrc
        echo 'export WORKON_HOME="${HOME}/.pyvirtualenvs"' >> ~/.virtenvrc
        echo 'source /usr/share/virtualenvwrapper/virtualenvwrapper_lazy.sh' >> ~/.virtenvrc
    fi
    if ! grep virtenvrc ~/.bashrc; then
        echo >> ~/.bashrc
        echo 'source ~/.virtenvrc' >> ~/.bashrc
    fi
    """)

    # bashrc is not sourced for non-interactive shells, so source the virtenvrc explicitly
    run("""
    export LC_ALL=C
    source ~/.virtenvrc
    if ! [ -d ${WORKON_HOME}/oggm_env ]; then
        mkvirtualenv oggm_env -p /usr/bin/python3
    fi
    """)


@task
def install_node_nfs_master(nn='', inst=None):
    """
    Setup the node to act as NFS server, serving /home and /work
    """
    if inst is None:
        inst = select_instance(nn)
    update_key_filename(inst.region.name)
    env.host_string = inst.dns_name
    env.user = 'ubuntu'

    run("""
    export LC_ALL=C &&
    export DEBIAN_FRONTEND=noninteractive &&
    sudo apt-get -y install nfs-kernel-server &&
    sudo mkdir -p /work/ubuntu /export/work /export/home &&
    sudo chown ubuntu:ubuntu /work/ubuntu &&
    echo '/export      *(rw,fsid=0,insecure,no_subtree_check,async)' > /tmp/exports &&
    echo '/export/work *(rw,nohide,insecure,no_subtree_check,async)' >> /tmp/exports &&
    echo '/export/home *(rw,nohide,insecure,no_subtree_check,async)' >> /tmp/exports &&
    sudo cp --no-preserve=all /tmp/exports /etc/exports &&
    cp /etc/fstab /tmp/fstab &&
    echo '/work /export/work none bind 0 0' >> /tmp/fstab &&
    echo '/home /export/home none bind 0 0' >> /tmp/fstab &&
    sudo cp --no-preserve=all /tmp/fstab /etc/fstab &&
    sudo mount /export/work &&
    sudo mount /export/home &&
    sudo sed -i 's/NEED_SVCGSSD=.*/NEED_SVCGSSD="no"/' /etc/default/nfs-kernel-server &&
    sudo service nfs-kernel-server restart &&
    echo "%s slots=$(( $(grep '^processor' /proc/cpuinfo | tail -n1 | cut -d ':' -f2 | xargs) + 1 ))" > /work/ubuntu/mpi_hostfile &&
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N "" &&
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys &&
    echo Done
    """ % inst.private_ip_address)


@task
def install_node_nfs_client(master_ip, nn='', inst=None):
    """
    Setup the node to act as NFS client on the given master_ip.
    """
    if inst is None:
        inst = select_instance(nn)
    update_key_filename(inst.region.name)
    env.host_string = inst.dns_name
    env.user = 'ubuntu'

    run("""
    export LC_ALL=C &&
    cd / &&
    sudo mkdir /work &&
    export DEBIAN_FRONTEND=noninteractive &&
    sudo apt-get -y install nfs-common &&
    cp /etc/fstab /tmp/fstab &&
    echo '%s:/work /work nfs4 _netdev,auto 0 0' >> /tmp/fstab
    echo '%s:/home /home nfs4 _netdev,auto 0 0' >> /tmp/fstab
    sudo cp --no-preserve=all /tmp/fstab /etc/fstab &&
    sudo mount /work &&
    echo "%s slots=$(( $(grep '^processor' /proc/cpuinfo | tail -n1 | cut -d ':' -f2 | xargs) + 1 ))" >> /work/ubuntu/mpi_hostfile &&
    echo Rebooting... && sleep 1 && sudo shutdown -r now
    """ % (master_ip, master_ip, inst.private_ip_address))


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


def select_instance(nn='', regions=def_regions):
    """
    Prompt the user to select an instance
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

    if nn == '' or nn is None:
        nn = prompt('Instance index:')

    nn = int(nn)

    if nn < 0 or nn >= len(instlist):
        print('Instance index out of range!')
        sys.exit(-1)

    return instlist[nn]


def select_volume(nn='', regions=def_regions):
    """
    Prompt the user to select a volume
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

    if nn == '' or nn is None:
        nn = prompt('Volume index:')

    nn = int(nn)

    if nn < 0 or nn >= len(vollist):
        print('Volume index out of range!')
        sys.exit(-1)

    return vollist[nn]


@task
def terminate_one(regions=def_regions, nn=''):
    """
    Terminate one instance
    """
    print('Select instance to terminate:')
    print()
    inst = select_instance(nn, regions)
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
    print('Select volume to terminate:')
    print()
    vol = select_volume(nn, regions)
    vol.delete()


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
def connect(nn='', user='ubuntu'):
    """
    SSH to cloud instances
    """
    inst = select_instance(nn)

    update_key_filename(inst.region.name)

    print('ssh', '-i', os.path.expanduser(env.key_filename), '%s@%s' % (user, inst.dns_name))
    print('...')
    print()

    os.execlp('ssh', 'ssh', '-i', os.path.expanduser(env.key_filename), '%s@%s' % (user, inst.dns_name))


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
    run("sudo mount -o defaults,discard /dev/xvdf1 /work")
    run("echo \"/dev/xvdf1 /work ext4 defaults,discard 0 0\" | sudo tee -a /etc/fstab")
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



