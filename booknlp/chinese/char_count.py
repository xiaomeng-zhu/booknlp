html_header = """<!DOCTYPE html>
<html>
<body>

"""

html_footer = """</body>
</html>"""

with open("examples/fengshou_chapter1.txt", "r") as f:
    fs_string = f.readlines()

fs_content = ""
index = 0
for line in fs_string:
    fs_content += "<p>"
    for char in line:
        if char.isspace() or char == "　": 
            fs_content += char
        else:
            fs_content += (char+"<sub>"+str(index)+"</sub>")
            index += 1
    fs_content += "</p>"

with open("fengshou_with_char.html", "w") as writer:
    all = html_header + fs_content + html_footer
    writer.write(all)

# huliyuan
with open("examples/with_poetry/huliyuanquanzhuan_chapter1.txt", "r") as f:
    hly_string = f.readlines()

hly_content = ""
index = 0
for line in hly_string:
    hly_content += "<p>"
    for char in line:
        if char.isspace() or char == "　": 
            hly_content += char
        else:
            hly_content += (char+"<sub>"+str(index)+"</sub>")
            index += 1
    hly_content += "</p>"


with open("huliyuan_with_char.html", "w") as writer:
    all = html_header + hly_content + html_footer
    writer.write(all)