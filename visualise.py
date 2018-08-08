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
    return code

def print_comment_and_code(content, p0, p1, p2):
    comment = ''
    code0 = '&nbsp;'
    code1 = '&nbsp;'
    code2 = '&nbsp;'
    if p0 is not None:
        part = content[0][p0]
        comment = [v[0] for v in part if v[0] is not None and v[1] is None]
        code0 = '\n'.join([v[1] for v in part if v[1] is not None])
        code0 = highlight(code0)
    if p1 is not None:
        part = content[1][p1]
        comment = [v[0] for v in part if v[0] is not None and v[1] is None]
        code1 = '\n'.join([v[1] for v in part if v[1] is not None])
        code1 = highlight(code1)
    if p2 is not None:
        part = content[2][p2]
        comment = [v[0] for v in part if v[0] is not None and v[1] is None]
        code2 = '\n'.join([v[1] for v in part if v[1] is not None])
        code2 = highlight(code2)

    class_name = ' shared-content'
    if p0 is not None and p1 is None and p2 is None:
        class_name = ' dynet'
    elif p0 is None and p1 is not None and p2 is None:
        class_name = ' pytorch'
    elif p0 is None and p1 is None and p2 is not None:
        class_name = ' tensorflow'

    print("""<div class="outer {}">""".format(class_name))

    print("""<div class="description{}">""".format(class_name), end="")
    if len(comment) > 0:
        print("\n<br />\n".join(comment), end="<br /><br />")
    else:
        print("&nbsp;")
    print("</div>")

    print("""<code class="tensorflow">""")
    print(code2, end="")
    print("</code>")
    empty_code1 = " empty" if len(comment) > 0 and code1 == "&nbsp;" and code2 != "&nbsp;" else ""
    print("""<code class="pytorch{}">""".format(empty_code1))
    print(code1, end="")
    print("</code>")
    empty_code0 = " empty" if len(comment) > 0 and code0 == "&nbsp;" else ""
    print("""<code class="dynet{}">""".format(empty_code0))
    print(code0, end="")
    print("</code>")

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

def match(part0, part1):
    part0 = ' '.join([v[1].strip() for v in part0 if v[1] is not None])
    part1 = ' '.join([v[1].strip() for v in part1 if v[1] is not None])
    return part0 == part1

def align(content):
    # Find parts in common between all three
    matches = []
    for i0, part0 in enumerate(content[0]):
        for i1, part1 in enumerate(content[1]):
            if match(part0, part1):
                for i2, part2 in enumerate(content[2]):
                    if match(part0, part2):
                        matches.append((i0, i1, i2))
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

head = """
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN"
   "http://www.w3.org/TR/html4/strict.dtd">

<html>
<head>
  <title></title>
  <meta http-equiv="content-type" content="text/html; charset=UTF-8">
  <style type="text/css">
body {
    background: #000000;
    color: #FFFFFF;
}
h1 {
    color: #EEEEEE;
    background: #111111;
    margin: 0px;
    text-align: center;
    padding: 10px;
}
div.buttons {
    width: 100%;
    display: flex;
    justify-content: center;
}
div.main {
    display: flex;
    flex-direction: column;
    align-items: center;     /* center items horizontally, in this case */
    padding-top: 10px;
}
div.header-outer {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: #111111;
    color: #EEEEEE;
    margin: 0px;
    padding: 10px;
    font-size: large;
}
div.disqus {
    max-width: 1000px;
    margin: auto;
}
div.header {
    max-width: 1000px;
}
div.outer {
    clear: both;
}
div.description {
    font-size: large;
    color: #36e6e8;
    text-align: justify;
    line-height: 112%;
    width: 400px;
    float: left;
}
code  {
    background: #000000;
    color: #FFFFFF;
    float: right;
    width: 100ch;
    padding-left: 15px;
}
code.empty {
    background: #666666;
    margin-top: 8px;
    max-height: 3px;
}
code.dynet {
}
code.pytorch {
}
code.tensorflow {
}
a {
    color: #00a1d6;
}
.button {
    background-color: #008CBA;
    border: 10px;
    color: white;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
}
td.linenos { background-color: #f0f0f0; padding-right: 10px; }
span.lineno { background-color: #f0f0f0; padding: 0 5px 0 5px; }
pre { line-height: 125%; font-family: Menlo, "Courier New", Courier, monospace; margin: 0 }
body .hll { background-color: #ffffcc }
body .c { color: #408080; } /* Comment */
body .err { border: 1px solid #FF0000 } /* Error */
body .k { color: #fceb60; } /* Keyword */
body .o { color: #FFFFFF } /* Operator */
body .ch { color: #36e6e8; } /* Comment.Hashbang */
body .cm { color: #36e6e8; } /* Comment.Multiline */
body .cp { color: #36e6e8 } /* Comment.Preproc */
body .cpf { color: #36e6e8; } /* Comment.PreprocFile */
body .c1 { color: #36e6e8; } /* Comment.Single */
body .cs { color: #36e6e8; } /* Comment.Special */
body .gd { color: #A00000 } /* Generic.Deleted */
body .ge { font-style: italic } /* Generic.Emph */
body .gr { color: #FF0000 } /* Generic.Error */
body .gh { color: #000080; } /* Generic.Heading */
body .gi { color: #00A000 } /* Generic.Inserted */
body .go { color: #888888 } /* Generic.Output */
body .gp { color: #000080; } /* Generic.Prompt */
body .gs { font-weight: bold } /* Generic.Strong */
body .gu { color: #800080; } /* Generic.Subheading */
body .gt { color: #0044DD } /* Generic.Traceback */
body .kc { color: #008000; } /* Keyword.Constant */
body .kd { color: #008000; } /* Keyword.Declaration */
body .kn { color: #84b0d9; } /* Keyword.Namespace */
body .kp { color: #008000 } /* Keyword.Pseudo */
body .kr { color: #008000; } /* Keyword.Reserved */
body .kt { color: #B00040 } /* Keyword.Type */
body .m { color: #BC94B7 } /* Literal.Number */
body .s { color: #BC94B7 } /* Literal.String */
body .na { color: #7D9029 } /* Name.Attribute */
body .nb { color: #36e6e8 } /* Name.Builtin */
body .nc { color: #36e6e8; } /* Name.Class */
body .no { color: #880000 } /* Name.Constant */
body .nd { color: #AA22FF } /* Name.Decorator */
body .ni { color: #999999; } /* Name.Entity */
body .ne { color: #D2413A; } /* Name.Exception */
body .nf { color: #36e6e8 } /* Name.Function */
body .nl { color: #A0A000 } /* Name.Label */
body .nn { color: #FFFFFF; } /* Name.Namespace */
body .nt { color: #008000; } /* Name.Tag */
body .nv { color: #19177C } /* Name.Variable */
body .ow { color: #fceb60; } /* Operator.Word */
body .w { color: #bbbbbb } /* Text.Whitespace */
body .mb { color: #BC94B7 } /* Literal.Number.Bin */
body .mf { color: #BC94B7 } /* Literal.Number.Float */
body .mh { color: #BC94B7 } /* Literal.Number.Hex */
body .mi { color: #BC94B7 } /* Literal.Number.Integer */
body .mo { color: #BC94B7 } /* Literal.Number.Oct */
body .sa { color: #BC94B7 } /* Literal.String.Affix */
body .sb { color: #BC94B7 } /* Literal.String.Backtick */
body .sc { color: #BC94B7 } /* Literal.String.Char */
body .dl { color: #BC94B7 } /* Literal.String.Delimiter */
body .sd { color: #BC94B7; } /* Literal.String.Doc */
body .s2 { color: #BC94B7 } /* Literal.String.Double */
body .se { color: #BB6622; } /* Literal.String.Escape */
body .sh { color: #BC94B7 } /* Literal.String.Heredoc */
body .si { color: #BB6688; } /* Literal.String.Interpol */
body .sx { color: #008000 } /* Literal.String.Other */
body .sr { color: #BB6688 } /* Literal.String.Regex */
body .s1 { color: #BC94B7 } /* Literal.String.Single */
body .ss { color: #19177C } /* Literal.String.Symbol */
body .bp { color: #36e6e8 } /* Name.Builtin.Pseudo */
body .fm { color: #36e6e8 } /* Name.Function.Magic */
body .vc { color: #FFFFFF } /* Name.Variable.Class */
body .vg { color: #FFFFFF } /* Name.Variable.Global */
body .vi { color: #FFFFFF } /* Name.Variable.Instance */
body .vm { color: #FFFFFF } /* Name.Variable.Magic */
body .il { color: #BC94B7 } /* Literal.Number.Integer.Long */
</style>
</head>

<body onload="toggleDyNet(); togglePyTorch(); toggleTensorflow()">

<h1>Implementing a neural Part-of-Speech tagger</h1>
<div class="header-outer">
<div class=header>
<p>
DyNet, PyTorch and Tensorflow are complex frameworks with different ways of approaching neural network implementation and different default behaviour.
This page is intended to show how to implement a non-trivial model in all three.
The design of the page is motivated by my own preference for an annotated complete piece of code, rather than introducing parts piecemeal with discussion.
For a non-tutorial version it would be better to use abstraction to improve flexibility, but that would have complicated the flow here.
</p>
<p>
Use the buttons to show one or more implementations and their associated comments.
Shared content is repeated and aligned.
Implementation-specific content is shown with empty space for the other implementations and lines to make the link from a comment to the code clear.
If you would like to copy large chunks of the code, go to <a href="https://github.com/jkkummerfeld/neural-tagger-tutorial">the repository</a> for this page.
</p>
<p>
The three implementations below all produce part-of-speech taggers that score ~97.2% on the development set of the Penn Treebank.
The specific hyperparameter choice follows <a href="https://arxiv.org/abs/1806.04470">Yang, Liang, and Zhang (CoLing 2018)</a> and matches their performance for the setting without a CRF or character-based word embeddings.
</p>
<p>
Making this helped me understand all three frameworks better. Hopefully you find it informative too!
</p>

<div class="buttons">
<button class="button" id="dybutton" onclick="toggleDyNet()">Hide DyNet</button>
<button class="button" id="ptbutton" onclick="togglePyTorch()">Hide PyTorch</button>
<button class="button" id="tfbutton" onclick="toggleTensorflow()">Hide Tensorflow</button>
</div>
</div>

</div>

"""

tail = """

<script>
function toggleDyNet() {
    var dyitems = document.getElementsByClassName("dynet");
    var dybutton = document.getElementById("dybutton");
    for (var i = dyitems.length - 1; i >= 0; i--)
    {
        if (dyitems[i].style.display === "none") {
            dyitems[i].style.display = "block";
            dybutton.textContent = "Hide DyNet";
            dybutton.style.backgroundColor = "#008CBA";
        } else {
            dyitems[i].style.display = "none";
            dybutton.textContent = "Show DyNet";
            dybutton.style.backgroundColor = "#4CAF50";
        }
    }
    toggleShared();
}
function togglePyTorch() {
    var pyitems = document.getElementsByClassName("pytorch");
    var ptbutton = document.getElementById("ptbutton");
    for (var i = pyitems.length - 1; i >= 0; i--)
    {
        if (pyitems[i].style.display === "none") {
            pyitems[i].style.display = "block";
            ptbutton.textContent = "Hide PyTorch";
            ptbutton.style.backgroundColor = "#008CBA";
        } else {
            pyitems[i].style.display = "none";
            ptbutton.textContent = "Show PyTorch";
            ptbutton.style.backgroundColor = "#4CAF50";
        }
    }
    toggleShared();
}
function toggleTensorflow() {
    var tfitems = document.getElementsByClassName("tensorflow");
    var tfbutton = document.getElementById("tfbutton");
    for (var i = tfitems.length - 1; i >= 0; i--)
    {
        if (tfitems[i].style.display === "none") {
            tfitems[i].style.display = "block";
            tfbutton.textContent = "Hide Tensorflow";
            tfbutton.style.backgroundColor = "#008CBA";
        } else {
            tfitems[i].style.display = "none";
            tfbutton.textContent = "Show Tensorflow";
            tfbutton.style.backgroundColor = "#4CAF50";
        }
    }
    toggleShared();
}
function toggleShared() {
    var dybutton = document.getElementById("dybutton");
    var ptbutton = document.getElementById("ptbutton");
    var tfbutton = document.getElementById("tfbutton");
    var allitems = document.getElementsByClassName("shared-content");
    if (dybutton.textContent === "Show DyNet" && tfbutton.textContent === "Show Tensorflow" && ptbutton.textContent === "Show PyTorch") {
        for (var i = allitems.length - 1; i >= 0; i--)
        {
            allitems[i].style.display = "none";
        }
    } else {
        for (var i = allitems.length - 1; i >= 0; i--)
        {
            allitems[i].style.display = "block";
        }
    }
}
</script>

<div class="header-outer">
<div class="header">
<p>
I developed this code with help from many people and resources. In particular:
<ul>
<li> <a href="https://github.com/jiesutd/NCRFpp">NCRFpp</a>, the code associated with <a href="https://arxiv.org/abs/1806.04470">Yang, Liang, and Zhang (CoLing 2018)</a>, which was my starting point for PyTorch and my reference point when trying to check performance for the others.</li>
<li> Members of the <a href="http://web.eecs.umich.edu/~wlasecki/croma.html">CROMA Lab</a> who gave feedback during development.</li>
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

