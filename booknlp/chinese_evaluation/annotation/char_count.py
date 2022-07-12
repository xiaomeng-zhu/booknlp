html_header = """<!DOCTYPE html>
<html>
<body>

"""

html_footer = """</body>
</html>"""

def generate_html(in_file_name, out_file_name):

    with open(in_file_name, "r") as f:
        string = f.readlines()

    content = ""
    index = 0
    for line in string:
        content += "<p>"
        for char in line:
            if char.isspace() or char == "　": 
                content += char
            else:
                content += (char+"<sub>"+str(index)+"</sub>")
                index += 1
        content += "</p>"

    with open(out_file_name, "w") as writer:
        all = html_header + content + html_footer
        writer.write(all)

if __name__ == "__main__":
    # generate_html("examples/fengshou_chapter1.txt", "annotation/fengshou_with_char.html")
    # generate_html("examples/with_poetry/huliyuan_chapter1.txt", "annotation/huliyuan_with_char.html")
    generate_html("examples/lu_xun/ah_q_chapter12.txt", "annotation/ahq_with_char.html")
    generate_html("examples/with_poetry/jinpingmei_chapter1.txt", "annotation/jinpingmei_with_char.html")
    generate_html("examples/with_poetry/niehaihua_excerpt.txt", "annotation/niehaihua_with_char.html")
    generate_html("examples/linglijiguang_chapter1.txt", "annotation/linglijiguang_with_char.html")