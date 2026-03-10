.. _installing-oggm-windows:

Installing OGGM on Windows
==========================

OGGM does not work on Windows. However, there is a workaround using the Windows
Subsystem for Linux (WSL). There is no official support for installing OGGM on
Windows. Nevertheless, the steps listed hereafter have been used for a successful
installation, but there is no guarantee that it will work for everyone.

Install WSL
-----------

Enable Intel Virtual Technology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a first step, you need to enable ‘Intel Virtual Technology’ to allow WSL to be
executed in the next step. To do this, restart your computer and access the BIOS or
UEFI settings by one of the following keys: F2, F12, Esc, Del. Which key you need
to use depends on your computer. With the arrow key you are now able to navigate to
the ‘configuration’ tab. Select ‘Intel Virtual Technology’ and change the settings
from ‘disabled’ to ‘enabled’. Save the change and exit the BIOS or UEFI settings by
using F10.

.. figure:: _static/windows_installation/enable_intel_virtual_technology.png
    :width: 100%


Enable WSL
~~~~~~~~~~

Next you need to enable WSL. This is possible through the Windows Features dialog
or Power Shell. In the Windows search bar, type 'features' to bring up the
**Turn Windows Features on or off** dialog. Scroll down and check **Windows
Subsystem for Linux**. It is also necessary to tick **Virtual Machine Platform**
and **Windows Hypervisor Platform** to avoid errors. Click ‘OK’ and in the prompted
new field ‘install’.

.. figure:: _static/windows_installation/windows_features_on_off.png
   :width: 40%

|

It is also possible to enable WSL as administrator via the Power Shell by using the
following command::

    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux

Afterwards you will be asked to restart Windows.

Check WSL
~~~~~~~~~

To be sure WSL is enabled you can check it by opening a Command Prompt and type ’WSL’
after restarting Windows:

.. figure:: _static/windows_installation/check_wsl.png
   :width: 50%

This means, WLS in enabled but you have not installed a Linux distribution yet.

Download Windows Subsystem for Linux and a Linux distribution of your choice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the free
`Windows subsystem for Linux <https://apps.microsoft.com/detail/windows-subsystem-for-linux/9P9TQF7MRM4R?hl=en-us&gl=US&ocid=pdpshare>`_
in the Microsoft Store. With that you keep Windows as main operating system and are able
work with Linux alongside your Windows application. There are some limitations compared
to a complete second operating system, but it is sufficient to install OGGM. In addition,
download and install `Ubuntu <https://apps.microsoft.com/detail/ubuntu/9PDXGNCFSCZV?hl=en-gb&gl=US>`_
or another Linux distribution of your choice.

UNIX User Account
~~~~~~~~~~~~~~~~~

Now open Ubuntu. You will automatically be asked to create a UNIX account.

.. figure:: _static/windows_installation/unix_user_account.png
   :width: 75%

Installing OGGM
---------------

After successfully setting up Linux as a Windows subsystem, you can proceed with the OGGM
installation process. There are multiple approaches available for
:doc:`installing-oggm`, and all of them should work with WSL.
