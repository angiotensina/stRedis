import pypandoc

class MarkdownToPDFConverter:
    def __init__(self, texto_markdown, outputfile, mainfont="Arial", papersize="letter", margin="1in"):
        """
        Inicializa la clase con el contenido en Markdown y los parámetros de conversión.
        """
        self.texto_markdown = texto_markdown
        self.outputfile = outputfile
        self.mainfont = mainfont
        self.papersize = papersize
        self.margin = margin

    def convertir_a_pdf(self):
        """
        Convierte el contenido en Markdown a PDF utilizando pypandoc y xelatex como motor.
        """
        try:
            output = pypandoc.convert_text(
                self.texto_markdown,
                'pdf',
                format='md',
                outputfile=self.outputfile,
                extra_args=[
                    '--pdf-engine=xelatex',
                    '-V', f'mainfont={self.mainfont}',
                    '-V', 'mathspec',
                    '-V', f'papersize={self.papersize}',
                    '-V', f'geometry:margin={self.margin}'
                ]
            )
            if output == "":
                print("Conversión a PDF completada con éxito.")
            else:
                print("Hubo un error durante la conversión.")
        except Exception as e:
            print(f"Error durante la conversión: {e}")