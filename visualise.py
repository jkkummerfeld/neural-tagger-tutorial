#!/usr/bin/env python3

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import sys

lexer = get_lexer_by_name("python", stripall=True)
formatter = HtmlFormatter(cssclass="source")

def print_comment_and_code(comment, code):
    code = highlight(code, lexer, formatter)
    if """<span></span><span class="o">.</span>""" in code:
        code = "".join(code.split("""<span></span><span class="o">.</span>"""))
    if len(comment.strip()) > 0:
        print("""<div class="outer">
      <div class="description">""", end="")
        print(comment, end="")
        print("</div><code>""")
        print(code, end="")
        print("</code></div>")
    else:
        print("""<div class="outer"><code>""")
        print(code, end="")
        print("</code></div>")

def main():
    code = sys.stdin.read()

    print(head)

    # Break into sections
    parts = [[]]
    for line in code.split("\n"):
        if len(line.strip()) == 0 or line.strip().startswith("####"):
            if len(parts[-1]) == 1:
                parts[-2].append(parts[-1].pop())
            else:
                parts.append([""])
        parts[-1].append(line)

    # Render
    for part in parts:
        comment = []
        code = []
        for line in part:
            if line.strip().startswith("####"):
                comment.append(line.strip()[4:].strip())
            else:
                code.append(line)
        print_comment_and_code("\n".join(comment), "."+ "\n".join(code))

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
code  {
    background: #000000;
    color: #FFFFFF;
    float: right;
    width: 70%;
    padding-left: 10px;
}
div.description {
    float: left;
    font-size: large;
    color: #36e6e8;
}
div.outer {
    width: 100%;
    clear: both;
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
body .bp { color: #008000 } /* Name.Builtin.Pseudo */
body .fm { color: #0000FF } /* Name.Function.Magic */
body .vc { color: #FFFFFF } /* Name.Variable.Class */
body .vg { color: #FFFFFF } /* Name.Variable.Global */
body .vi { color: #FFFFFF } /* Name.Variable.Instance */
body .vm { color: #FFFFFF } /* Name.Variable.Magic */
body .il { color: #BC94B7 } /* Literal.Number.Integer.Long */
</style>
</head>

<body>
"""

tail = """</body>
</html>"""

if __name__ == '__main__':
    main()

