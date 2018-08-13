#!/usr/bin/env python3

import sys

import pygments
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

lexer = get_lexer_by_name("python", stripall=True)
formatter = HtmlFormatter(cssclass="source")

def highlight(raw_code):
    code = pygments.highlight(raw_code, lexer, formatter)
    if len(raw_code) > 0:
        if raw_code[-1] == '\n':
            code = code.split("</pre></div>")[0] +"\n</pre></div>"
        if raw_code[0] == ' ':
            indent = 0
            for i, char in enumerate(raw_code):
                if char != ' ':
                    break
                indent += 1
            parts = code.split("<span class")
            parts[0] += ' '*indent
            code = "<span class".join(parts)
    if code.startswith("""<div class="source">""") and code.endswith("</div>"):
        code = code[20:-6]
    return code

def print_comment_and_code(content, p0, p1, p2):
    comment0 = ''
    comment1 = ''
    comment2 = ''
    code0 = '&nbsp;'
    code1 = '&nbsp;'
    code2 = '&nbsp;'
    if p0 is not None:
        part = content[0][p0]
        comment0 = [v[0] for v in part if v[0] is not None and v[1] is None]
        code0 = '\n'.join([v[1] for v in part if v[1] is not None])
        code0 = highlight(code0)
    if p1 is not None:
        part = content[1][p1]
        comment1 = [v[0] for v in part if v[0] is not None and v[1] is None]
        code1 = '\n'.join([v[1] for v in part if v[1] is not None])
        code1 = highlight(code1)
    if p2 is not None:
        part = content[2][p2]
        comment2 = [v[0] for v in part if v[0] is not None and v[1] is None]
        code2 = '\n'.join([v[1] for v in part if v[1] is not None])
        code2 = highlight(code2)

    class_name = 'shared-content'
    comment = comment0
    if p0 is not None and p1 is None and p2 is None:
        if len(''.join(comment).strip()) > 0:
            class_name = 'dynet'
    elif p0 is None and p1 is not None and p2 is None:
        comment = comment1
        if len(''.join(comment).strip()) > 0:
            class_name = 'pytorch'
    elif p0 is None and p1 is None and p2 is not None:
        comment = comment2
        if len(''.join(comment).strip()) > 0:
            class_name = 'tensorflow'

    print("""<div class="outer {}">""".format(class_name))
    
    div_comment = """<span class="description {}">""".format(class_name)
    if len(''.join(comment).strip()) > 0:
        div_comment += "\n<br />\n".join(comment) +"<br /><br />"
    else:
        div_comment += "&nbsp;"
    div_comment += "</span>"

    if len(''.join(comment).strip()) > 0:
        if code0 == "&nbsp;":
            if code1 == "&nbsp;":
                order = 'tfonly'
            else:
                order = 'ptonly'
    
    div_code0 = """<code class="dynet">""" + code0 +"</code>"
    div_code1 = """<code class="pytorch">""" + code1 +"</code>"
    div_code2 = """<code class="tensorflow">""" + code2 + "</code>"
    if class_name == 'pytorch':
        div_code0 = """<code class="dynet pytorch-line">""" + code0 +"</code>"
        div_code1 = """<code class="pytorch">""" + code1 +"</code>"
        div_code2 = """<code class="tensorflow">""" + code2 + "</code>"
    elif class_name == 'tensorflow':
        div_code0 = """<code class="dynet tensorflow-line">""" + code0 +"</code>"
        div_code1 = """<code class="pytorch tensorflow-line">""" + code1 +"</code>"
        div_code2 = """<code class="tensorflow">""" + code2 + "</code>"

    print(div_comment, end="")
    print(div_code0, end="")
    print(div_code1, end="")
    print(div_code2, end="")

    print("</div>")

def read_file(filename):
    parts = [[]]
    prev_comment = True
    for line in open(filename):
        line = line.strip('\n')
        
        # Update comment status
        if line.strip().startswith("####"):
            if not prev_comment:
                parts.append([])
            prev_comment = True
        else:
            prev_comment = False

        # Divide up the line
        comment = None
        code = None
        if line.strip().startswith("####"):
            comment = line.strip()[4:].strip()
        elif '#' in line and line.strip()[0] != '#':
            comment = line.split("#")[-1]
            code = line[:-len(comment)-1]
            comment = comment.strip()
        else:
            code = line

        parts[-1].append((comment, code))
    return parts

def match(part0, part1, do_comments=False):
    if do_comments:
        part0 = ' '.join([v[0].strip() for v in part0 if v[0] is not None and v[1] is None])
        part1 = ' '.join([v[0].strip() for v in part1 if v[0] is not None and v[1] is None])
        return part0 == part1 and part0.strip() != ''
    else:
        part0 = ' '.join([v[1].strip() for v in part0 if v[1] is not None])
        part1 = ' '.join([v[1].strip() for v in part1 if v[1] is not None])
        return part0 == part1

def align(content):
    # Find parts in common between all three
    matches = set()
    for i0, part0 in enumerate(content[0]):
        for i1, part1 in enumerate(content[1]):
            if match(part0, part1):
                for i2, part2 in enumerate(content[2]):
                    if match(part0, part2):
                        matches.add((i0, i1, i2))
            if match(part0, part1, True):
                for i2, part2 in enumerate(content[2]):
                    if match(part0, part2, True):
                        matches.add((i0, i1, i2))
    matches = sorted(list(matches))
    return matches

def main():
    # Read data
    content = [read_file(filename) for filename in sys.argv[1:]]

    # Work out aligned sections
    matches = align(content)

    # Render
    print(head)

    print("""<div class="main">""")
    positions = [0 for _ in content]
    for p0, p1, p2 in matches:
        while positions[0] < p0:
            print_comment_and_code(content, positions[0], None, None)
            positions[0] += 1
        while positions[1] < p1:
            print_comment_and_code(content, None, positions[1], None)
            positions[1] += 1
        while positions[2] < p2:
            print_comment_and_code(content, None, None, positions[2])
            positions[2] += 1
        print_comment_and_code(content, p0, p1, p2)
        positions[0] += 1
        positions[1] += 1
        positions[2] += 1

    print("""<br /></div>""")

    print(tail)

###style_dark = """
###  <style type="text/css">
###body {
###    background: #000000;
###    color: #FFFFFF;
###}
###h1 {
###    color: #EEEEEE;
###    background: #111111;
###    margin: 0px;
###    text-align: center;
###    padding: 10px;
###}
###div.buttons {
###    width: 100%;
###    display: flex;
###    justify-content: center;
###}
###div.main {
###    display: flex;
###    flex-direction: column;
###    align-items: center;     /* center items horizontally, in this case */
###    padding-top: 10px;
###}
###div.header-outer {
###    display: flex;
###    flex-direction: column;
###    align-items: center;
###    background: #111111;
###    letter-spacing: 1px;
###    line-height: 130%;
###    color: #e0e0e0;
###    margin: 0px;
###    padding: 10px;
###    font-size: large;
###}
###div.disqus {
###    max-width: 1000px;
###    margin: auto;
###}
###div.header {
###    max-width: 1000px;
###}
###div.outer {
###    clear: both;
###}
###div.description {
###    letter-spacing: 1px;
###    font-size: large;
###    color: #97cae0;
###    line-height: 112%;
###    width: 400px;
###    float: left;
###}
###code  {
###    background: #000000;
###    color: #FFFFFF;
###    float: right;
###    width: 100ch;
###    padding-left: 15px;
###}
###code.empty {
###    background: #666666;
###    margin-top: 8px;
###    max-height: 3px;
###}
###code.dynet {
###}
###code.pytorch {
###}
###code.tensorflow {
###}
###a {
###    color: #00a1d6;
###}
###.button {
###    cursor: pointer;
###    background-color: #008CBA;
###    border: 10px;
###    margin: 5px;
###    color: white;
###    padding: 15px 32px;
###    text-align: center;
###    text-decoration: none;
###    display: inline-block;
###    font-size: 16px;
###}
###td.linenos { background-color: #f0f0f0; padding-right: 10px; }
###span.lineno { background-color: #f0f0f0; padding: 0 5px 0 5px; }
###pre { line-height: 125%; font-family: Menlo, "Courier New", Courier, monospace; margin: 0 }
###body .hll { background-color: #ffffcc }
###body .c { color: #408080; } /* Comment */
###body .err { border: 1px solid #FF0000 } /* Error */
###body .k { color: #fceb60; } /* Keyword */
###body .o { color: #FFFFFF } /* Operator */
###body .ch { color: #36e6e8; } /* Comment.Hashbang */
###body .cm { color: #36e6e8; } /* Comment.Multiline */
###body .cp { color: #36e6e8 } /* Comment.Preproc */
###body .cpf { color: #36e6e8; } /* Comment.PreprocFile */
###body .c1 { color: #36e6e8; } /* Comment.Single */
###body .cs { color: #36e6e8; } /* Comment.Special */
###body .gd { color: #A00000 } /* Generic.Deleted */
###body .ge { font-style: italic } /* Generic.Emph */
###body .gr { color: #FF0000 } /* Generic.Error */
###body .gh { color: #000080; } /* Generic.Heading */
###body .gi { color: #00A000 } /* Generic.Inserted */
###body .go { color: #888888 } /* Generic.Output */
###body .gp { color: #000080; } /* Generic.Prompt */
###body .gs { font-weight: bold } /* Generic.Strong */
###body .gu { color: #800080; } /* Generic.Subheading */
###body .gt { color: #0044DD } /* Generic.Traceback */
###body .kc { color: #008000; } /* Keyword.Constant */
###body .kd { color: #008000; } /* Keyword.Declaration */
###body .kn { color: #84b0d9; } /* Keyword.Namespace */
###body .kp { color: #008000 } /* Keyword.Pseudo */
###body .kr { color: #008000; } /* Keyword.Reserved */
###body .kt { color: #B00040 } /* Keyword.Type */
###body .m { color: #BC94B7 } /* Literal.Number */
###body .s { color: #BC94B7 } /* Literal.String */
###body .na { color: #7D9029 } /* Name.Attribute */
###body .nb { color: #36e6e8 } /* Name.Builtin */
###body .nc { color: #36e6e8; } /* Name.Class */
###body .no { color: #880000 } /* Name.Constant */
###body .nd { color: #AA22FF } /* Name.Decorator */
###body .ni { color: #999999; } /* Name.Entity */
###body .ne { color: #D2413A; } /* Name.Exception */
###body .nf { color: #36e6e8 } /* Name.Function */
###body .nl { color: #A0A000 } /* Name.Label */
###body .nn { color: #FFFFFF; } /* Name.Namespace */
###body .nt { color: #008000; } /* Name.Tag */
###body .nv { color: #19177C } /* Name.Variable */
###body .ow { color: #fceb60; } /* Operator.Word */
###body .w { color: #bbbbbb } /* Text.Whitespace */
###body .mb { color: #BC94B7 } /* Literal.Number.Bin */
###body .mf { color: #BC94B7 } /* Literal.Number.Float */
###body .mh { color: #BC94B7 } /* Literal.Number.Hex */
###body .mi { color: #BC94B7 } /* Literal.Number.Integer */
###body .mo { color: #BC94B7 } /* Literal.Number.Oct */
###body .sa { color: #BC94B7 } /* Literal.String.Affix */
###body .sb { color: #BC94B7 } /* Literal.String.Backtick */
###body .sc { color: #BC94B7 } /* Literal.String.Char */
###body .dl { color: #BC94B7 } /* Literal.String.Delimiter */
###body .sd { color: #BC94B7; } /* Literal.String.Doc */
###body .s2 { color: #BC94B7 } /* Literal.String.Double */
###body .se { color: #BB6622; } /* Literal.String.Escape */
###body .sh { color: #BC94B7 } /* Literal.String.Heredoc */
###body .si { color: #BB6688; } /* Literal.String.Interpol */
###body .sx { color: #008000 } /* Literal.String.Other */
###body .sr { color: #BB6688 } /* Literal.String.Regex */
###body .s1 { color: #BC94B7 } /* Literal.String.Single */
###body .ss { color: #19177C } /* Literal.String.Symbol */
###body .bp { color: #36e6e8 } /* Name.Builtin.Pseudo */
###body .fm { color: #36e6e8 } /* Name.Function.Magic */
###body .vc { color: #FFFFFF } /* Name.Variable.Class */
###body .vg { color: #FFFFFF } /* Name.Variable.Global */
###body .vi { color: #FFFFFF } /* Name.Variable.Instance */
###body .vm { color: #FFFFFF } /* Name.Variable.Magic */
###body .il { color: #BC94B7 } /* Literal.Number.Integer.Long */
###</style>
###"""

style_light = """
  <style type="text/css">
body {
    padding: 0px;
    margin: 0px;
}
h1 {
    margin: 0px;
    text-align: center;
    padding: 10px;
}
div.buttons {
    width: 100%;
    display: flex;
    justify-content: center;
}
div.header-outer {
    background: #f3f3f3;
    display: flex;
    flex-direction: column;
    align-items: center;
    line-height: 130%;
    margin: 0px;
    padding: 10px;
    font-size: large;
}
div.header {
    max-width: 1000px;
}
button {
    cursor: pointer;
    margin: 5px;
    border: none;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    outline: none;
}
#dybutton {
    background-color: #f8fad2;
    border: 3px solid #f8fad2;
}
#tfbutton {
    background-color: #d0eaf0;
    border: 3px solid #d0eaf0;
}
#ptbutton { 
    background-color: #fbe1d3;
    border: 3px solid #fbe1d3;
}
div.disqus {
    max-width: 1000px;
    margin: auto;
}

div.main {
    text-align: center;
    padding-top: 10px;
}
div.outer {
    white-space: nowrap;
    display: none;
}
span.description {
    text-align: left;
    vertical-align: top;
    font-size: large;
    line-height: 112%;
    width: 400px;
    white-space: normal;
    display: none;
}
span.description.dynet {
    background-color: #f8fad2;
    display: none;
}
span.description.pytorch {
    background-color: #fbe1d3;
    display: none;
}
span.description.tensorflow {
    background-color: #d0eaf0;
    display: none;
}
span.paragraph-start {
    font-weight: bold;
}
code {
    text-align: left;
    vertical-align: top;
    width: 100ch;
    display: none;
    padding-left: 5px;
    padding-right: 5px;
}
code.pytorch-line {
    display: inline-block;
    background: #fbe1d3;
    margin-top: 8px;
    max-height: 3px;
}
code.tensorflow-line {
    display: inline-block;
    background: #d0eaf0;
    margin-top: 8px;
    max-height: 3px;
}
pre { }
td.linenos { background-color: #f0f0f0; padding-right: 10px; }
span.lineno { background-color: #f0f0f0; padding: 0 5px 0 5px; }
pre { line-height: 125%; font-family: Menlo, "Courier New", Courier, monospace; margin: 0 }
body .bp { color: #0a5ac2 } /* Name.Builtin.Pseudo */
body .c1 { color: #6a727c } /* Comment.Single */
body .fm { color: #0a5ac2 } /* Name.Function.Magic */
body .k { color: #d63e4d } /* Keyword */
body .kn { color: #d63e4d; } /* Keyword.Namespace */
body .mf { color: #0a5ac2 } /* Literal.Number.Float */
body .mi { color: #0a5ac2 } /* Literal.Number.Integer */
body .n { color: #000000 } /* Variable */
body .nb { color: #0a5ac2 } /* Name.Builtin */
body .nc { color: #6f40bf } /* Name.Class */
body .nf { color: #6f40bf } /* Name.Function */
body .nn { color: #000000 } /* Name.Namespace */
body .o { color: #d63e4d } /* Operator */
body .ow { color: #d63e4d } /* Operator.Word */
body .p { color: #000000 } /* Parentheses */
body .s1 { color: #052e60 } /* Literal.String.Single */
body .s2 { color: #052e60 } /* Literal.String.Double */
body .sd { color: #052e60 } /* Literal.String.Doc */
body .vm { color: #0a5ac2 } 

body .c { color: #999988 } /* Comment */
body .cm { color: #6a727c } /* Comment.Multiline */
body .cp { color: #6a727c } /* Comment.Preproc */
body .cs { color: #6a727c } /* Comment.Special */
body .err { color: #a61717 } /* Error */
body .gd { color: #000000 } /* Generic.Deleted */
body .ge { color: #000000 } /* Generic.Emph */
body .gh { color: #999999 } /* Generic.Heading */
body .gi { color: #000000 } /* Generic.Inserted */
body .go { color: #888888 } /* Generic.Output */
body .gp { color: #555555 } /* Generic.Prompt */
body .gr { color: #aa0000 } /* Generic.Error */
body .gs { } /* Generic.Strong */
body .gt { color: #aa0000 } /* Generic.Traceback */
body .gu { color: #aaaaaa } /* Generic.Subheading */
body .hll { }
body .kc { color: #000000; } /* Keyword.Constant */
body .kd { color: #000000; } /* Keyword.Declaration */
body .kp { color: #000000; } /* Keyword.Pseudo */
body .kr { color: #000000; } /* Keyword.Reserved */
body .kt { color: #445588; } /* Keyword.Type */
body .m { color: #009999 } /* Literal.Number */
body .no { color: #008080 } /* Name.Constant */
body .s { color: #d01040 } /* Literal.String */
body .na { color: #008080 } /* Name.Attribute */
body .nd { color: #3c5d5d } /* Name.Decorator */
body .ni { color: #800080 } /* Name.Entity */
body .ne { color: #990000 } /* Name.Exception */
body .nl { color: #990000 } /* Name.Label */
body .nt { color: #000080 } /* Name.Tag */
body .nv { color: #008080 } /* Name.Variable */
body .w { color: #bbbbbb } /* Text.Whitespace */
body .mh { color: #0a5ac2 } /* Literal.Number.Hex */
body .mo { color: #0a5ac2 } /* Literal.Number.Oct */
body .sb { color: #052e60 } /* Literal.String.Backtick */
body .sc { color: #052e60 } /* Literal.String.Char */
body .se { color: #052e60 } /* Literal.String.Escape */
body .sh { color: #052e60 } /* Literal.String.Heredoc */
body .si { color: #052e60 } /* Literal.String.Interpol */
body .sx { color: #052e60 } /* Literal.String.Other */
body .sr { color: #052e60 } /* Literal.String.Regex */
body .ss { color: #052e60 } /* Literal.String.Symbol */
body .vc { color: #008080 } /* Name.Variable.Class */
body .vg { color: #008080 } /* Name.Variable.Global */
body .vi { color: #008080 } /* Name.Variable.Instance */
body .il { color: #009999 } /* Literal.Number.Integer.Long */
</style>
"""

head = """<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN"
   "http://www.w3.org/TR/html4/strict.dtd">
<html>
<head>
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-19179423-2"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-19179423-2');
</script>
  <title>Neural Tagger Implementations</title>
  <meta http-equiv="content-type" content="text/html; charset=UTF-8">
"""+ style_light +"""
</head>

<body>
<div class="header-outer">
<h1>Implementing a neural Part-of-Speech tagger</h1>
<h3>by Jonathan K. Kummerfeld <a href="https://www.jkk.name/">[site]</a></h3>
<div class=header>
<p>
DyNet, PyTorch and Tensorflow are complex frameworks with different ways of approaching neural network implementation and variations in default behaviour.
This page is intended to show how to implement the same non-trivial model in all three.
The design of the page is motivated by my own preference for a complete program with annotations, rather than the more common tutorial style of introducing code piecemeal in between discussion.
The design of the code is also geared towards providing a complete picture of how things fit together.
For a non-tutorial version of this code it would be better to use abstraction to improve flexibility, but that would have complicated the flow here.
</p>
<p>
<span class="paragraph-start">Model:</span>
The three implementations below all define a part-of-speech tagger with word embeddings initialised using GloVe, fed into a one-layer bidirectional LSTM, followed by a matrix multiplication to produce scores for tags.
They all score ~97.2% on the development set of the Penn Treebank.
The specific hyperparameter choices follows <a href="https://arxiv.org/abs/1806.04470">Yang, Liang, and Zhang (CoLing 2018)</a> and matches their performance for the setting without a CRF layer or character-based word embeddings.
The <a href="https://github.com/jkkummerfeld/neural-tagger-tutorial">repository</a> for this page provides the code in runnable form.
The only dependencies are the respective frameworks (DyNet <a href="https://github.com/clab/dynet/releases/tag/2.0.3">2.0.3</a>, PyTorch <a href="https://github.com/pytorch/pytorch/releases/tag/v0.4.1">0.4.1</a> and Tensorflow <a href="https://github.com/tensorflow/tensorflow/releases/tag/v1.9.0">1.9.0</a>).
</p>
<p>
<span class="paragraph-start">Website usage:</span> Use the buttons to show one or more implementations and their associated comments (note, depending on your screen size you may need to scroll to see all the code).
Matching or closely related content is aligned.
Framework-specific comments are highlighted in a colour that matches their button and a line is used to make the link from the comment to the code clear.
</p>
<p>
Making this helped me understand all three frameworks better. Hopefully you will find it informative too!
</p>

<div class="buttons">
<button class="button" id="dybutton" onmouseover="" onclick="toggleDyNet()">Show/Hide DyNet</button>
<button class="button" id="ptbutton" onmouseover="" onclick="togglePyTorch()">Show/Hide PyTorch</button>
<button class="button" id="tfbutton" onmouseover="" onclick="toggleTensorflow()">Show/Hide Tensorflow</button>
</div>
</div>

</div>

"""

tail = """

<script>
var dyShowing = false;
var tfShowing = false;
var ptShowing = false;
function whichShowing() {
    if (dyShowing && tfShowing && ptShowing) return "all";
    else if (dyShowing && ptShowing) return "-t";
    else if (dyShowing && tfShowing) return "-p";
    else if (ptShowing && tfShowing) return "-d";
    else if (dyShowing) return "d";
    else if (ptShowing) return "p";
    else if (tfShowing) return "t";
    else return "-";
}
function toggleItem(toEdit, showing) {
    if (
        (showing === "d" && toEdit.classList.contains("dynet")) ||
        (showing === "p" && toEdit.classList.contains("pytorch")) ||
        (showing === "t" && toEdit.classList.contains("tensorflow")) ||
        (showing === "-d" && toEdit.classList.contains("pytorch")) ||
        (showing === "-p" && toEdit.classList.contains("dynet")) ||
        (showing === "-t" && toEdit.classList.contains("dynet")) ||
        (showing === "-d" && toEdit.classList.contains("tensorflow")) ||
        (showing === "-p" && toEdit.classList.contains("tensorflow")) ||
        (showing === "-t" && toEdit.classList.contains("pytorch")) ||
        (showing != "-" && toEdit.classList.contains("shared-content")) ||
        showing === "all"
       ) {
        if (toEdit.classList.contains('outer')) {
            toEdit.style.display = "block";
        } else {
            toEdit.style.display = "inline-block";
        }
    } else {
        toEdit.style.display = "none";
    }
}
function toggleDyNet() {
    dyShowing = ! dyShowing;

    var dybutton = document.getElementById("dybutton");
    if (dyShowing) {
        dybutton.style.backgroundColor = "#f1f5a4";
        dybutton.style.border = "3px solid #777777";
    } else {
        dybutton.style.backgroundColor = "#f8fad2";
        dybutton.style.border = "3px solid #f8fad2";
    }

    toggleAll();
}
function togglePyTorch() {
    ptShowing = ! ptShowing;

    var ptbutton = document.getElementById("ptbutton");
    if (ptShowing) {
        ptbutton.style.backgroundColor = "#f7c1a4";
        ptbutton.style.border = "3px solid #777777";
    } else {
        ptbutton.style.backgroundColor = "#fbe1d3";
        ptbutton.style.border = "3px solid #fbe1d3";
    }

    toggleAll();
}
function toggleTensorflow() {
    tfShowing = ! tfShowing;

    var tfbutton = document.getElementById("tfbutton");
    if (tfShowing) {
        tfbutton.style.backgroundColor = "#a9d9e4";
        tfbutton.style.border = "3px solid #777777";
    } else {
        tfbutton.style.backgroundColor = "#d0eaf0";
        tfbutton.style.border = "3px solid #d0eaf0";
    }

    toggleAll();
}
function toggleAll() {
    showing = whichShowing();
    var dyitems = document.getElementsByClassName("dynet");
    for (var i = dyitems.length - 1; i >= 0; i--) {
        toggleItem(dyitems[i], showing);
    }
    var tfitems = document.getElementsByClassName("tensorflow");
    for (var i = tfitems.length - 1; i >= 0; i--) {
        toggleItem(tfitems[i], showing);
    }
    var ptitems = document.getElementsByClassName("pytorch");
    for (var i = ptitems.length - 1; i >= 0; i--) {
        toggleItem(ptitems[i], showing);
    }
    var allitems = document.getElementsByClassName("shared-content");
    for (var i = allitems.length - 1; i >= 0; i--) {
        toggleItem(allitems[i], showing);
    }
}
</script>

<div class="header-outer">
<div class="header">
<p>
A few miscellaneous notes:
<ul>
    <li>PyTorch 0.4 does not support recurrent dropout directly. For an example of how to achieve it, see the LSTM and QRNN Language Model Toolkit's <a href="https://github.com/salesforce/awd-lstm-lm/blob/28683b20154fce8e5812aeb6403e35010348c3ea/weight_drop.py">WeightDrop class</a> and <a href="https://github.com/salesforce/awd-lstm-lm/blob/457a422eb46e970a6aad659ca815a04b3d074d6c/model.py#L22">how it is used</a>.</li>
    <li>Tensorflow 1.9 does not support weight decay directly, but <a href="https://github.com/tensorflow/tensorflow/pull/17438">this pull request</a> appears to add support and will be part of 1.10.</li>
</ul>
</p>
<p>
I developed this code and webpage with help from many people and resources. In particular:
<ul>
    <li> Feedback from <a href="http://proebsting.cs.arizona.edu/">Todd Proebsting</a>, <a href="http://www.it.usyd.edu.au/~judy/">Judy Kay</a>, <a href="http://www.it.usyd.edu.au/~bob/">Bob Kummerfeld</a>, and members of the <a href="http://web.eecs.umich.edu/~wlasecki/croma.html">CROMA Lab</a>.</li>
    <li> <a href="https://github.com/jiesutd/NCRFpp">NCRFpp</a>, the code associated with <a href="https://arxiv.org/abs/1806.04470">Yang, Liang, and Zhang (CoLing 2018)</a>, which was my starting point for PyTorch and my reference point when trying to check performance for the others.</li>
    <li> Guillaume Genthial's blog post about <a href="https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html">Sequence Tagging with Tensorflow</a>. </li>
    <li> The DyNet <a href="https://github.com/clab/dynet/blob/master/examples/tagger/bilstmtagger.py">example tagger</a>. </li>
</ul>
</p>
</div>
</div>

<div class="disqus">
<div id="disqus_thread"></div>
</div>
<script>
var disqus_config = function () {
    this.page.url = 'http://jkk.name/neural-tagger-tutorial/';
    this.page.identifier = '/neural-tagger-tutorial/';
    this.page.title = 'Neural Tagger Example';
};
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://www-jkk-name.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
</body>
</html>"""

if __name__ == '__main__':
    main()

