
class Propaganda_Techniques():


    TECHNIQUE_NAMES_FILE="data/propaganda-techniques-names.txt"

    def __init__(self, filename=TECHNIQUE_NAMES_FILE):

        with open(filename, "r") as f:
             self.techniques = [ line.rstrip() for line in f.readlines() ]


    def is_valid_technique(self, technique_name):

        return technique_name in self.techniques


    def __str__(self):

        return ",".join(self.techniques)


    #def load_technique_names_from_file(filename=TECHNIQUE_NAMES_FILE):
    #with open(filename, "r") as f:
    #    return [ line.rstrip() for line in f.readlines() ]

