#!---------written by Felix Oesterle (FSO)-----------------
#-DESCRIPTION:
# This is based on fabfile from Raincloud Project (simplified)
#
#-Last modified:  Thu Jul 09, 2015  13:10
#@author Felix Oesterle 
#-----------------------------------------------------------
from __future__ import with_statement
from fabric.api import *
import boto.ec2
import os
import time
import sys
import socket
import re
import fileinput
import shutil
import datetime
import math
from collections import defaultdict

#-----------------------------------------------------------
# SHORT DOCU 
#-----------------------------------------------------------

# -------- SETUP BOTO and Fabric-----------------
# pip install boto
# pip install fabric
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
#     7. Once you have enough, shut the instances down with 
#         fab cloud_terminate


#-----------------------------------------------------------
# SETUP
#-----------------------------------------------------------
env.disable_known_hosts=True
env.user = 'root'

# FSO--- default name used in tags and instance names:
# set this eg. to your name
def_cn = 'AWS'
 
# FSO--- ssh and credentials setup
# FSO---the name of the amazon keypair (will be created if it does not exist) 
keyn='ec2_OGGR_FS' 
# FSO--- the same name as you used in boto setup XXXX (see Readme)
ec2Profile = 'oggr'
def_key_dir='~/.ssh'

# FSO--- Amazon AWS region setup
def_regions = ['us-east-1','eu-west-1'] #regions for spot search
def_default_avz = 'us-east-1b' #Default availability zone if ondemand is used

# FSO--- type of instance pricing, either:
# ondemand: faster availability, more expensive
# spot: cheaper, takes longer to start up, might be shutdown without warning
# def_default_requesttype = 'spot' #
def_default_requesttype = 'ondemand' #

# FSO--- the AMI to use 
def_ami = dict()
def_ami['eu'] = 'ami-47a23a30' #eu Ubuntu 14.04 LTS
# def_ami['us'] = 'ami-5189a661' #us Ubuntu 14.04 LTS
def_ami['us'] = 'ami-d05e75b8' #us Ubuntu 14.04 LTS

# FSO---instance types 
# always uncomment matching def_itype and vcpus (TODO: change to dict)
# def_itype = 't2.micro' # 2ct
# vcpus = 1
def_itype = 'm3.xlarge' 
vcpus = 4
# def_itype = 'c4.2xlarge' 
# vcpus = 8
# def_itype = 'c4.8xlarge' 
# vcpus = 32 


# FSO---log file with timestamps to analyse cloud performance 
# look at it with tail -f cloudexecution.log
def_logfile='/Users/felixs/data/AWS/cloudexecution.log'


#-----------------------------------------------------------
# SETUP END
#-----------------------------------------------------------

ec2_prices = dict()
ec2_prices['t2.micro'] = 0.014
ec2_prices['m3.xlarge'] = 0.308
ec2_prices['c4.2xlarge'] = 0.528 
env.key_filename = def_key_dir+'/'+keyn+'.pem' 
def_price = ec2_prices[def_itype] 


@task
def cloud_make(cn=def_cn,nn=0):
    """
    Start and prepare instance  -THIS IS THE MAIN ACTIVITY-
    """
    t = time.time()
    log_with_ts("fabric started ------------------------------")
    log_with_ts("Instance: " + def_itype+ "("+str(vcpus)+" CPUs)")
    
    # FSO---set best avz 
    if def_default_requesttype == 'spot':
        best_avz,request_type = get_cheapest_availability_zone(def_price)
    else:
        best_avz = def_default_avz
        request_type = 'ondemand'
        
    print best_avz, request_type
    log_with_ts('avz: ' + best_avz)
    log_with_ts('request_type: ' + request_type)
 
    # FSO--- start instances 
    instance_start(cn=cn,avz=best_avz,rt=request_type)
    print("Done setting up instance")
    log_with_ts("instance ready")

    t_init = time.time() 

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
    
    
    t_end = time.time() 

    print "Time needed for init (min)", (t_init - t)/60.
    print "Time needed for workflow and terminate", (t_end - t_init)/60.
    log_with_ts("fabric end")

@task
def instance_start(cn=def_cn,ami=def_ami,inst_type=def_itype, \
        avz=def_default_avz,
        rt=def_default_requesttype):
    """
    Start and prepare instances 
    """
    # FSO---find already existing nodes 
    cloud = boto.ec2.connect_to_region(avz[:-1],profile_name=ec2Profile)
    filters = {'tag:type': cn+'node'}
    insta = cloud.get_all_instances(filters =filters)
    
    # FSO---install each new node
    print ( "Requesting new instance")
    log_with_ts("Requesting  new instance")


    nodenumber = len(insta) + 1
 
    node_install(cn=cn,avz=avz,rt=rt,idn=nodenumber)
    
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
        print "Connection to globus"
        
    log_with_ts("workflow done")
    print "Time needed for run_workflow (min)", (time.time()-t)/60.0

    #copy result
    # FSO--- TODO need to check decisions in agwl file to get correct files 
    target_file = '~/tmp/screenlog.0'
    try:
        get('~/screenlog.0',target_file)
    except:
        print "Uh oh, no screenlog file found"
        log_with_ts("no transfer of results")

    return target_file

@task
def cloud_list(cn=def_cn,itype='all',regions=def_regions):
    """
    List all ec2 instances.
    """
    for region in regions:
        cloud = boto.ec2.connect_to_region(region,profile_name=ec2Profile)
        instances = cloud.get_all_instances()
        vols = cloud.get_all_volumes()
        print 
        print "-------CURRENT RUNNING-----------"
        print "       REGION:",region
        
        update_costs(cn=cn,regions=regions,itype=itype)

        for reservation in instances:
            for inst in reservation.instances:
                if inst.state != 'terminated':
                    cu_time = datetime.datetime.utcnow()
                    it =  datetime.datetime.strptime(inst.launch_time,'%Y-%m-%dT%H:%M:%S.000Z') 
                else:
                    try:
                        cu_time =datetime.datetime.strptime(inst.tags.get('terminate_time'),'%Y-%m-%dT%H:%M:%S.%f') 
                    except:
                        cu_time = datetime.datetime.utcnow()

                    it =  datetime.datetime.strptime(inst.launch_time,'%Y-%m-%dT%H:%M:%S.000Z') 
                    
                time_taken = cu_time - it
                hours, rest = divmod(time_taken.total_seconds(),3600)
                minutes, seconds = divmod(rest, 60)
                    
                print inst.id, inst.instance_type, \
                            inst.tags.get('Name'), \
                            inst.tags.get('type'), \
                            inst.state,\
                            inst.dns_name,\
                          inst.private_ip_address,\
                          inst.private_dns_name,\
                            inst.tags.get('current_price'), \
                            inst.tags.get('billable_hours'), \
                            inst.tags.get('terminate_time'), \
                            inst.placement
                #print inst.__dict__.keys()
                # FSO--- -1 is a dirty fix for different timezones 
                print "running for: ", hours,'h', minutes, "min"
        for vol in vols:
            print vol.id,"\t", vol.status,'\t',vol.size

@task
def node_install(cn=def_cn,inst_type=def_itype,idn=0, \
        avz=def_default_avz,rt=def_default_requesttype,
        group_name='oggmssh',
        ssh_port=22,
        cidr='0.0.0.0/0'):
    """
    Request and prepare single instance
    """
    # FSO---connect  
    cloud = boto.ec2.connect_to_region(avz[:-1],profile_name=ec2Profile)
    aminfo = cloud.get_image(def_ami[avz[0:2]])
    key_dir = def_key_dir
    
    # FSO---check if node with same name alread exists
    if node_exists(cn+'_node'+str(idn)):
        print("Node already exists")
        sys.exit()
    
    # FSO---create a bigger root device 
    # TODO: bring this to settings at beginning
    dev_sda1 = boto.ec2.blockdevicemapping.EBSBlockDeviceType()
    dev_sda1.size = 50 # size in Gigabytes
    bdm = boto.ec2.blockdevicemapping.BlockDeviceMapping()
    bdm['/dev/sda1'] = dev_sda1 

    # Check to see if specified keypair already exists.
    # If we get an InvalidKeyPair.NotFound error back from EC2,
    # it means that it doesn't exist and we need to create it.
    try:
        key = cloud.get_all_key_pairs(keynames=[keyn])[0]
    except cloud.ResponseError, e:
        if e.code == 'InvalidKeyPair.NotFound':
            print 'Creating keypair: %s' % keyn
            # Create an SSH key to use when logging into instances.
            key = cloud.create_key_pair(keyn)

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

    # Check to see if specified security group already exists.
    # If we get an InvalidGroup.NotFound error back from EC2,
    # it means that it doesn't exist and we need to create it.
    try:
        group = cloud.get_all_security_groups(groupnames=[group_name])[0]
    except cloud.ResponseError, e:
        if e.code == 'InvalidGroup.NotFound':
            print 'Creating Security Group: %s' % group_name
            # Create a security group to control access to instance via SSH.
            group = cloud.create_security_group(group_name, \
                    'A group that allows SSH access')
        else:
            raise

    # Add a rule to the security group to authorize SSH traffic
    # on the specified port.
    try:
        group.authorize('tcp', ssh_port, ssh_port, cidr)
    except cloud.ResponseError, e:
        if e.code == 'InvalidPermission.Duplicate':
            print 'Security Group: %s already authorized' % group_name
        else:
            raise
 
    log_with_ts("request node "+str(idn))
    print 'Reserving instance for node', aminfo.id, inst_type, aminfo.name, aminfo.region 
    
    if rt == 'spot':
        print ("placing node in ",avz)
        requests = cloud.request_spot_instances(def_price, 
                      def_ami[avz[0:2]], 
                      count=1,
                      type='one-time', 
                      security_groups=[group_name],
                      key_name=keyn, 
                      placement=avz,
                      instance_type=inst_type,
                      block_device_map=bdm)
        req_ids = [request.id for request in requests]
        instance_ids = wait_for_fulfillment(cloud,req_ids)
        instances = cloud.get_only_instances(instance_ids=instance_ids)
        node = instances[0]
        log_with_ts("fullfilled spot node "+str(idn))
    else:
        print ("placing node in ",avz)
        reservation = cloud.run_instances(image_id=def_ami[avz[0:2]], 
                key_name=keyn, 
                placement = avz,
                security_groups=[group_name],
                instance_type=inst_type,
                block_device_map= bdm)
        node = reservation.instances[0]
        log_with_ts("fullfilled ondemand node "+str(idn))
    
    time.sleep(2)
    while not node.update() == 'running':
        print 'waiting for ',cn,'node ',idn,' to boot...'
        time.sleep(5)

    log_with_ts("booted node "+str(idn))

    node.add_tag('Name', cn+'_node'+str(idn))
    node.add_tag('type', cn+'node')
#     
    # FSO---set delete on termination flag to true for ebs block device 
    node.modify_attribute('blockDeviceMapping', { '/dev/sda1' : True }) 
#     
    # FSO--- test socket connect to ssh service
    ssh_test(node)
    log_with_ts("reachable node "+str(idn))
# 
    # FSO---enable root (also sets env.host_name needed for fab to run run())
    enable_root(node.dns_name) 
    env.user = 'root'

    # FSO---install needed packages 
    # INSTALL YOUR SOFTWARE HERE TODO: put in external file
    # if def_ami[avz[0:2]] == 'ami-640a0610' or def_ami[avz[0:2]] == 'ami-edb0ec84':
#         #For AMAZON Ireland ami-640a0610(Ubuntu 12.10)o
#         #For AMAZON US ami-a73264ce(Ubuntu 12.10)o
#         with hide('stdout'):
#             run('apt-get -u update; sleep 4',shell=True, pty=True)
#             run('apt-get -y install libnetcdf-dev',shell=True, pty=True)
#             run('wget http://netcdf4-python.googlecode.com/files/netCDF4-1.0.4.tar.gz')
#             run('tar -xvzf netCDF4-1.0.4.tar.gz')
#             with cd('~/netCDF4-1.0.4'):
#                 run('python setup.py install')
#             # FSO---not necessary, but helps during debugging
#             run('apt-get -y install speedometer')
#     else:
#         print 'Dont know how to prepare that imagae'
#         sys.exit()
#     run('mkdir -p ~/.gridarm')
#     put('./node_files/'+aska_app+'.tar.gz','~/')
#     run('tar xvzf '+aska_app+'.tar.gz')
#     with settings(warn_only=True):
#         run('ln -s /usr/bin/python /root/'+rc_scriptdir+'/python ')
    log_with_ts("finished node "+str(idn))

@task
def cloud_terminate(cn=def_cn,itype='all',regions=def_regions):
    """
    Terminate all instances
    """
    print regions
    for region in regions:
        print 
        print "-------CURRENT RUNNING-----------"
        print "       REGION:",region

        cloud = boto.ec2.connect_to_region(region,profile_name=ec2Profile)
        instances = cloud.get_all_instances()
        vol = cloud.get_all_volumes()
        
        update_costs(cn=cn,itype=itype)
        
        for reservation in instances:
            for inst in reservation.instances:
                if inst.state != 'terminated':
                    if itype == 'all':
                        print 'TERMINATING', inst.tags.get('Name'), inst.dns_name
                        inst.add_tag('Name', 'term')
                        inst.add_tag('type', 'term')
                        inst.terminate()
                        stati2 = datetime.datetime.utcnow()
                        inst.add_tag('terminate_time', stati2.isoformat())
                    elif itype == 'node' and inst.tags.get('type') == cn+'node':
                        print 'TERMINATING', inst.tags.get('Name'), inst.dns_name
                        inst.add_tag('Name', 'term')
                        inst.add_tag('type', 'term')
                        inst.terminate()
                        stati2 = datetime.datetime.utcnow()
                        inst.add_tag('terminate_time', stati2.isoformat())
                    elif itype == 'master' and inst.tags.get('type') == cn+'master':
                        print 'TERMINATING', inst.tags.get('Name'), inst.dns_name
                        inst.add_tag('Name', 'term')
                        inst.add_tag('type', 'term')
                        inst.terminate()
                        stati2 = datetime.datetime.utcnow()
                        inst.add_tag('terminate_time', stati2.isoformat())

        for unattachedvol in vol:
            if unattachedvol.status == 'available':
                print unattachedvol.id,"\t", unattachedvol.status, "... deleted"
                unattachedvol.delete()
            else:
                print unattachedvol.id,"\t", unattachedvol.status, "... not deleted"
 
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
        print 
        print "----------REGION:",region,itype,'-----------'

        
        for reservation in instances:
            for inst in reservation.instances:
                if inst.state == 'running' and (inst.tags.get('type')==cn+itype or itype=='all'):
                    hours = float(inst.tags.get('billable_hours'))
                    cu_price = float(inst.tags.get('current_price'))
                    cu_ondemand_price = hours * ec2_prices[inst.instance_type]
                    
                    print 
                    print inst.id, inst.instance_type, \
                                inst.tags.get('Name'), \
                                inst.dns_name,\
                                inst.tags.get('current_price')+'USD', \
                                inst.tags.get('billable_hours')+'h', \
                                inst.placement
                    # print 'Billable hours ',hours
                    # print 'Current price', cu_price
                    # print 'Current ondemand price', cu_ondemand_price
                    costs['ondemand'] += cu_ondemand_price
                    if inst.spot_instance_request_id == None:
                        print 'ondemand instance'
                        costs['running'] = cu_ondemand_price
                    else:
                        print 'spot instance'
                        costs['running'] += cu_price

    print 
    print 'Total ondemand: ', costs['ondemand'] 
    print 'Total of running: ' , costs['running']

    return costs

@task
def connect(cn=def_cn,nn='',avz='all'):
    """
    SSH to cloud instances (as root)
    """
    instlist = list()
    i = 0
    # cloud = boto.ec2.connect_to_region(def_default_avz[:-1])
    for region in def_regions:
        cloud = boto.ec2.connect_to_region(region,profile_name=ec2Profile)
        instances = cloud.get_all_instances()
        print
        for reservation in instances:
            for inst in reservation.instances:
                if inst.state == 'running':
                    print i, inst.tags.get('Name'),inst.dns_name, inst.private_ip_address, inst.placement
                    instlist.append(inst.dns_name)
                    i = i + 1
    
    print
    if nn =='':
        nn = prompt('Which instance to connect to ?:')
    
    print 'connecting to', instlist[int(nn)]

    sshcmd = 'ssh -i '+env.key_filename+' root@'+instlist[int(nn)]
    local(sshcmd) 

# @task
# def cloud_clean(cn=def_cn,avz=def_default_avz):
#     """
#     Clean master and nodes for next workflow run
#     """
#     # FSO---clean master  
#     master = node_find(cn+'_master',avz=avz)
#     env.host_string=master.dns_name
#     env.user = 'root'
# 
#     with settings(warn_only=True):
#         run('rm ~/.gridarm/nodes/*')
#         run('rm ~/.gridarm/glare/activity/types/raincloud*') 
#         run('rm ~/.gridarm/glare/activity/deployments/raincloud*') 
# 
#     # FSO---find existing nodes 
#     cloud = boto.ec2.connect_to_region(avz[:-1])
#     filters = {'tag:type': cn+'node'}
#     insta = cloud.get_all_instances(filters =filters)
#     for reservation in insta:
#         for inst in reservation.instances:
#             env.host_string=inst.dns_name
#             print 'Cleaning node: ',inst.dns_name
#             with settings(warn_only=True):
#                 run('rm -rf EE2-*')
# 

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
            if price.instance_type == def_itype:
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
    
    best_avz_nodes= min(maxprice_nodes, key=maxprice_nodes.get) # gets just the first if serveral avz's are the same

    print( "Cheapest nodes: ", best_avz_nodes, maxprice_nodes[best_avz_nodes] )
    print( "Ondemand nodes (EU):", ondemand_price )
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
                print "spot request `{}` fulfilled!".format(request.id)
                #print request.__dict__
                instances.append(request.instance_id)
                cloud.cancel_spot_instance_requests(request.id)
            elif request.state == 'cancelled':
                pending_ids.pop(pending_ids.index(request.id))
                print "spot request `{}` cancelled!".format(request.id)
            else:
                print "waiting on `{}`".format(request.id)
        time.sleep(5)
    print "all spots fulfilled!"
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
                    if inst.spot_instance_request_id != None:
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
                        total_price = hours * ec2_prices[inst.instance_type]
                
                    inst.add_tag('current_price',total_price )
                    inst.add_tag('billable_hours',hours )

def log_with_ts(logtext="no text given",lf=def_logfile):
    """
    Helper function to write logs with timestamps
    """
    logtime = time.time()
#    st = datetime.datetime.fromtimestamp(logtime).strftime('%Y-%m-%d %H:%M:%S')
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
#            start_time=stati.isoformat(),
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
                print 'found', inst.tags.get('Name'), inst.dns_name
                return inst

def node_exists(node,avz=def_default_avz):
    """
    checks if node with given name exists
    """
    exists = False
    cloud = boto.ec2.connect_to_region(avz[:-1],profile_name=ec2Profile)
    instances = cloud.get_all_instances()
    for reservation in instances:
        for inst in reservation.instances:
            if inst.tags.get('Name') == node and inst.state == 'running':
                print 'found', inst.tags.get('Name'), inst.dns_name
                exists = True
    
    return exists
               
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
            print 'waiting for ssh daemon...'
            time.sleep(5)
        finally:
            sock.close()



