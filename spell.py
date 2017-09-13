import random

material = {
    'pwd':'print working directory',
    "hostname": "my computers network name",
    'mkdir':'make directory',
    'cd':'change directory',
    'ls': 'list directory',
    'rmdir':'remove directory',
    'pushd':'push directory',
    'popd':'pop directory',
    'cp':'copy a file or directory',
    'robocopy':'robust copy',
    'mv':'move a file or directory',
    'more':'page through a file',
    'type':'print the whole file',
    'forfiles':'run a command on lots of files',
    'dir -r':'find files',
    'select-string':'find things inside files',
    'help':'read a manual page',
    'helpctr':'find what man page is appropriate',
    'echo':'print some arguments',
    'set':'export/set a new environment variable',
    'exit':'exit the shell',
    'runas':'DANGER! become super user root DANGER!'
}

print(list(material.keys()))

testing = True

while testing:
    test_key = random.choice(list(material.keys()))
    test_input = input("\'" + material[test_key] + "\'" + " corresponds to which cmd?")
    if test_input == 'q':
        testing = False
    elif test_input == 'l':
        print(list(material.keys()))
    elif test_input in material and material[test_input] == material[test_key]:
        print("Great!")
        #testing = True if (input("Another one? (y/n)") == 'y') else False
    else:
        print("That is just plain wrong. Should be \'" + test_key + "\'.")

print("Thank you! I hope you learned something today!")

#hade varit nice att inte få de en redan svarat rätt på. Skulle kunna ha lite statistik på det. Ev en dict med keys
# från material, vars value är en int och plussas på varje gång en svarar rätt. Så borde randomiseringen vara biased
# utifrån hur hög score en har för varje key: hög score innebär en lägre sannolikhet för att väljas ut,
# låg score innebär högre sannolikhet för att väljas ut.
# Oklart med implementationen av det, bara. 