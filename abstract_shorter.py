"""
Python script for removing the last entity of a triple and putting ""@en in stead. Used to shorten long abstract file.
"""

import sys

file = sys.argv[1]

reading_file = open(file, "r")

new_file_content = ""
i = 0
for i,line in enumerate(reading_file):
  if line.startswith("#"):
    continue
  stripped_line = line.strip()
  stripped_line = stripped_line.split(" ")
  new_line = stripped_line[0] + " " + stripped_line[1] + " \"" +str(i)+"\"@en ."
  i+=1
  new_file_content += new_line +"\n"
reading_file.close()

writing_file = open("long_abstract_en_mod.nt", "w")
writing_file.write(new_file_content)
writing_file.close()