import typst
# Compile `hello.typ` to PDF and save as `hello.pdf`
typst.compile("hello.typ", output="hello.pdf")
print("hello.pdf has been generated.")