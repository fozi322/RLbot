# Note
You can uncompile `.pyc` files using [PYLingual](https://pylingual.io). For a tutorial on how to do this, check out [this YouTube video](https://www.youtube.com/watch?v=lx91kgjHYwc) by PyLingual.

# Usage
Uncompile the `main.pyc` file and modify it as needed—whether you want to crack it, make adjustments, or rebrand the bot. After uncompiled using PYLingual, you can run the script by opening the command line and typing `py main.py`.

# RLbot
Kernelcheats.cc Rocket League bot source  
[+] Extremely advanced AI

This AI outperforms the existing ones. 

============================

**NOTE:** This AI requires a powerful PC. We recommend trying a daily key first to test your system's capability. Minimum requirements are:
- 8 CPU cores
- 16 GB RAM
- A strong GPU

Current versions and achievable ranks:
- V2 Max Rank: Grand Champion II, Division 3
- V3 Max Rank: Grand Champion II, Division 1
- V4 Max Rank: Bronze I - Supersonic Legend

# Another Thing
After uncompiled, you can recompile it using `python -m compileall your_file.py`. To convert the project, which includes `.pyc` files, folders, and subfolders, into an executable, you can use the following methods:

### Method 1: PyInstaller
1. Install PyInstaller: `pip install pyinstaller`
2. Navigate to your project directory in the command line.
3. Run PyInstaller with your main script: `pyinstaller --onefile main.py`
4. This will create a single executable file in the `dist` folder within your project directory.

### Method 2: cx_Freeze
1. Install cx_Freeze: `pip install cx_Freeze`
2. Create a `setup.py` file with the following content:
   ```python
   from cx_Freeze import setup, Executable

   setup(
       name="YourAppName",
       version="1.0",
       description="Description of your application",
       executables=[Executable("main.py")]
   )
3. Run the setup script: `python setup.py build`
4. The executable will be generated in the `build` folder within your project directory.
### Method 3: py2exe (Windows only)
1. Install py2exe: `pip install py2exe`
2. Create a `setup.py` file with the following content:
   ```python
   from distutils.core import setup
   import py2exe

   setup(console=['main.py'])
3. Run the setup script: `python setup.py py2exe`
4. The executable will be generated in the `dist` folder within your project directory.
