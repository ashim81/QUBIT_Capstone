"""
    Scanner and Parser for the grammar in grammarQBits.txt
 
    Token classes
    -------------
    STATE     '|' [0|1] '>'                  e.g. |0110>
    FUNCNAME  [a-zA-Z]([a-zA-Z]|[0-9])+     (at least 2 characters, per the grammar)
    IMG       [0-9]+ 'i'                     e.g. 3i
    REAL      [0-9]+                         e.g. 42
    PLUS '+'  MINUS '-'  STAR '*'  SLASH '/'
    LPAREN '('  RPAREN ')'  COMMA ','

"""

import re

"""
    @class Token
    @brief Represents an individual token of a string

    @details This class contains the string value of the token, the type of token it is, and what line it was found on

    @author Wyatt Fischer
    @version 1.0.0
    @date 2026-10-30
"""
class Token:
    type: str
    svalue: str
    linenum : int

    def __init__(self, type, svalue, linenum):
        self.type = type
        self.svalue = svalue
        self.linenum = linenum

    """
        @brief Creates a string representation of the token

        @returns String object with the values of the token
    """
    def __str__(self):
        return f"{self.type} {self.svalue} line: {self.linenum}\n"

"""
    @class Node
    @brief Represents a single Node of the AST

    @details Contains a function call to a printNode function that raises an error

    @author Wyatt Fischer
    @version 1.0.0
    @date 2026-10-30
"""
class Node:
    def printNode(self, indent = 0):
        raise NotImplementedError

"""
    @class Op
    @brief Subclass of the Node class that represents an operation

    @details Contains the string of the operand and left and right children nodes of the operator. As well as a print function

    @author Wyatt Fischer
    @version 1.0.0
    @date 2026-10-30
"""
class Op(Node):
    op: str # +, -, *, /
    left: Node
    right: Node

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    """
        @brief prints the operator node of the AST and its children nodes

        @returns A string object with the values of this node and its children
    """
    def printNode(self, indent = 0):
        padding = ".  "  * indent
        return (f"{padding}Operator: {self.op}\n"
                f"{self.left.printNode(indent + 1)}\n"
                f"{self.right.printNode(indent + 1)}")

"""
    @class State
    @brief Represents a basis state vector

    @details This class contains the string of bits representing the basis vector and a print function 

    @author Wyatt Fischer
    @version 1.0.0
    @date 2026-10-30
"""
class State(Node):
    s_bits: str # 0, 1

    def __init__(self, s_bits):
        self.s_bits = s_bits

    """
        @brief Prints the State vector value of the AST Node

        @returns String Object with the state vector value
    """
    def printNode(self, indent = 0):
        padding = ".  "  * indent
        return f"{padding}State: |{self.s_bits}>"


"""
    @class Num
    @brief Represents a number either a real or imaginary component

    @details This class contains the string of the component and the numerical value and a print function

    @author Wyatt Fischer
    @version 1.0.0
    @date 2026-10-30
"""
class Num(Node):
    kind: str # real or img
    value: int

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    """
        @brief Prints the Num node of the AST

        @returns String Object with the numerical value of the node
    """
    def printNode(self, indent = 0):
        padding = ".  "  * indent
        return f"{padding}Num ({self.kind}): {self.value}{"i" if self.kind == "img" else ""})"


"""
    @class Call
    @brief Represents a function call

    @details This class contains the name of the function call, a list of arguments that are passed in, and a print function

    @author Wyatt Fischer
    @version 1.0.0
    @date 2026-10-30
"""
class Call(Node):
    name: str
    args: List[Node]

    def __init__(self, name, args):
        self.name = name
        self.args = args

    """
        @brief Prints the Call node of the AST and its argument list

        @return String Object with the value of the function call and its arguments
    """
    def printNode(self, indent = 0):
        padding = ".  "  * indent
        if not self.args:
            return f"{padding}Call: {self.name}"
        inner = "\n".join(arg.printNode(indent + 1) for arg in self.args)
        return f"{padding}Call: {self.name}\n{inner}"
    
class ScannerError(Exception):
    pass

class ParseError(Exception):
    pass

TOKEN_REGEX = [
    ("STATE", r"\|[0|1]>"),
    ("IMG", r"[0-9]*i"),
    ("FUNCNAME", r"[a-zA-z][a-zA-Z0-9]*"),
    ("REAL", r"[0-9]+"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MUL", r"\*"),
    ("DIV", r"/"),
    ("OPENPAREN", r"\("),
    ("CLOSEPAREN", r"\)"),
    ("COMMA", r",")
]

#Define master regex with group captures
_MASTER = re.compile("|".join(f"(?P<{token}>{regex})" for token, regex in TOKEN_REGEX))
_WHITESPACE = re.compile(r"\s+")


"""
    @brief Scans a given string input breaking it into reconized tokens as defined by TOKEN_REGEX

    @param text The given string input
"""
def scan(text):
    pos, line = 0, 1
    n = len(text)

    while pos < n:
        #Get Whitespace section
        ws = _WHITESPACE.match(text, pos)
        if ws:
            #increment line num
            ws_segment = ws.group()
            nl_count = ws_segment.count("\n")
            if nl_count:
                line += nl_count
            pos = ws.end()

        #Get Matched Token
        token = _MASTER.match(text, pos)
        if not token:
            raise ScannerError(
                f"Illegal cahracter {text[pos]} at line: {line}"
            )
        #yield created token
        yield Token(token.lastgroup, token.group(), line)
        pos = token.end()

"""
    @brief Turns a given string input into a list of regonized tokens

    @param text The text to break into tokens

    @returns A list of tokens
"""
def tokenize(text):
    return list(scan(text))

"""
    @class Parser
    @brief A recursive decent parser that reconizes the language defined in grammarQBits.txt

    @details Contains several node helper functions and a parse function that analysis a given input

    @author Wyatt Fischer
    @version 1.0.0
    @date 2026-10-30
"""
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.index = 0

    """
        @brief Looks at the top of the token list and returns the top token
        @returns A Token
    """
    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    """
        @brief Looks at the top of the token list and returns the type of top token
        @returns A Token Type
    """
    def peek_type(self):
        tok = self.peek()
        return tok.type if tok else None

    """
        @brief Increments index to the next token in the token list
        @returns The previously indexed token
    """
    def advance(self) -> Token:
        tok = self.peek()
        self.index += 1
        return tok

    """
        @brief Looks at the top of the token list and returns the type of top token
        @returns A Token Type
    """
    def expect(self, ttype: str) -> Token:
        if self.peek_type() != ttype:
            self.error(ttype)
        return self.advance()

    """
        @brief Raises an error stating that the next token does not match the expected next token
    """
    def error(self, expected: str):
        tok = self.peek()
        if tok is None:
            raise ParseError(f"Unexpected end of input, expected {expected}")
        raise ParseError(f"Unexpected {tok.type} {tok.svalue} at line {tok.linenum}, expected {expected}")

    """
        @brief Parses the list of tokens
        @returns The head of the AST
    """
    def parse(self) -> Node:
        node = self.exp()
        if self.peek() is not None:
            self.error("end of input")
        return node

    """
        @brief Represents the exp productions
        @returns The exp node
    """
    def exp(self) -> Node:
        node = self.mulexp()
        while self.peek_type() in ("PLUS", "MINUS"):
            op = self.advance().svalue
            node = Op(op, node, self.mulexp())
        return node

    """
        @brief Represents the mulexp productions
        @returns The mulexp node
    """
    def mulexp(self) -> Node:
        node = self.id()
        while self.peek_type() in ("MUL", "DIV"):
            op = self.advance().svalue
            node = Op(op, node, self.id())
        return node

    """
        @brief Represents the id productions
        @returns The id node
    """
    def id(self) -> Node:
        t = self.peek_type()
        if t == "STATE":
            svalue = self.advance().svalue
            return State(svalue[1:-1]) #remove |, >
        if t == "FUNCNAME":
            return self.call()
        if t == "REAL":
            return Num("real", int(self.advance().svalue))
        if t == "IMG":
            svalue = self.advance().svalue
            if len(svalue) == 1: #if imganiary is i
                return Num("img", 1) 
            return Num("img", int(svalue[:-1])) #remove i
        if t == "LPAREN":
            self.advance()
            node = self.exp()
            self.expect("RPAREN")
            return node
        self.error("STATE, FUNCNAME, REAL, IMG, or '('")

    """
        @brief Represents the call productions
        @returns The Call node
    """
    def call(self) -> Node:
        name = self.expect("FUNCNAME").svalue
        self.expect("LPAREN")
        args = self.args()
        self.expect("RPAREN")
        return Call(name, args)

    """
        @brief Creates the list of nodes, including an empty list
        @returns A list of Nodes
    """
    def args(self) -> List[Node]:
        if self.peek_type() == "RPAREN":
            return []
        return self.arg_list()

    """
        @brief Creates the list of nodes that represent the argument list
        @returns A List of exp Nodes
    """
    def arg_list(self) -> List[Node]:
        result = [self.exp()]
        while self.peek_type() == "COMMA":
            self.advance
            result.append(self.exp())
        return result