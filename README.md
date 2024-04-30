# How to setup locally

1. Ensure anaconda or miniconda are installed, instructions are here for all platforms https://docs.anaconda.com/free/miniconda/miniconda-install/
2. Ensure VSCode is installed and setup.
3. Clone this repository where you would like it to live using:
   `git clone git@github.com:Yummi305/mushroomHarvest.git`
4. Open the new folder created from above in VSCode.
5. When opening the folder, install the recommended VSCode extensions (should be presented as an option).
6. In VSCode, Press 'CMD + Shift + P' (MAC) or 'CTRL + Shift + P' (Windows) and search for 'Python: Create Environment'. Select this.
7. Select 'Conda'.
8. If presented with "Delete and Recreate" and "Use Existing" select "Delete and Recreate" (you will not see these options if this is the first time you have run it).
9. Select Python V3.8
10. This will setup the environment with the packages defined in the 'environment.yml' file.
11. To test this has worked, in VSCode Press 'CMD + Shift + P' (MAC) or 'CTRL + Shift + P' (Windows) and search for/select "Terminal: Create New Terminal (In Active Workspace)".
12. In the terminal, run the command:
    `python main.py`
13. Verify that there are no errors and the top 5 rows of the csv file in `data/mushrooms.csv` is displayed.
