r"/home/meghana/Downloads/BL9000-Owners-Manual.pdf"
r"/home/meghana/Downloads/MP2531CE-1CA-Owners-Manual-1.pdf"

import re
from pathlib import Path
from typing import List, Dict, Any
from mistralai import Mistral
from mistralai.models import DocumentURLChunk


class PDFToMarkdownProcessor:
    """
    A class to process PDF files using Mistral OCR and extract table of contents
    from the resulting markdown.
    """

    def __init__(self, api_key: str):
        """
        Initialize the processor with Mistral API key.

        Args:
            api_key (str): Mistral API key
        """
        self.mistral_client = Mistral(api_key=api_key)

    def pdf_to_markdown(self, pdf_path: Path) -> str:
        """
        Convert PDF to markdown using Mistral OCR.

        Args:
            pdf_path (Path): Path to the PDF file

        Returns:
            str: Markdown content extracted from PDF
        """
        try:
            # Upload file to Mistral
            uploaded_file = self.mistral_client.files.upload(
                file={"file_name": pdf_path.stem, "content": pdf_path.read_bytes()},
                purpose="ocr"
            )

            # Get signed URL for processing
            signed_url = self.mistral_client.files.get_signed_url(
                file_id=uploaded_file.id,
                expiry=1
            )

            # Process PDF with OCR
            pdf_response = self.mistral_client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=False
            )

            # Extract markdown from response
            ocr_response = pdf_response.model_dump()
            markdown_content = "\n".join([
                page.get("markdown", "") for page in ocr_response.get("pages", [])
            ])

            return markdown_content

        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def extract_table_of_contents(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Extract table of contents from markdown content.

        Args:
            markdown_content (str): Markdown content

        Returns:
            List[Dict]: List of headings with their levels and text
        """
        toc = []

        # Regex pattern to match markdown headings (# ## ### etc.)
        heading_pattern = r'^(#{1,6})\s+(.+)$'

        lines = markdown_content.split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            match = re.match(heading_pattern, line)

            if match:
                hash_marks = match.group(1)
                heading_text = match.group(2).strip()
                level = len(hash_marks)

                # Clean up heading text (remove any markdown formatting)
                clean_text = re.sub(r'\*\*(.+?)\*\*', r'\1', heading_text)  # Remove bold
                clean_text = re.sub(r'\*(.+?)\*', r'\1', clean_text)  # Remove italic
                clean_text = re.sub(r'`(.+?)`', r'\1', clean_text)  # Remove code
                clean_text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', clean_text)  # Remove links

                toc.append({
                    'level': level,
                    'text': clean_text,
                    'raw_text': heading_text,
                    'line_number': line_num
                })

        return toc

    def generate_toc_markdown(self, toc: List[Dict[str, Any]]) -> str:
        """
        Generate a formatted table of contents in markdown format.

        Args:
            toc (List[Dict]): Table of contents data

        Returns:
            str: Formatted table of contents in markdown
        """
        if not toc:
            return "No headings found in the document."

        toc_lines = ["# Table of Contents", ""]

        for item in toc:
            indent = "  " * (item['level'] - 1)
            # Create anchor link (convert to lowercase, replace spaces with hyphens)
            anchor = re.sub(r'[^\w\s-]', '', item['text'].lower())
            anchor = re.sub(r'[-\s]+', '-', anchor).strip('-')

            toc_lines.append(f"{indent}- [{item['text']}](#{anchor})")

        return "\n".join(toc_lines)

    def print_toc_structure(self, toc: List[Dict[str, Any]]) -> None:
        """
        Print the table of contents structure in a readable format.

        Args:
            toc (List[Dict]): Table of contents data
        """
        if not toc:
            print("No headings found in the document.")
            return

        print("Document Structure:")
        print("=" * 50)

        for item in toc:
            indent = "  " * (item['level'] - 1)
            level_indicator = "H" + str(item['level'])
            print(f"{indent}{level_indicator}: {item['text']} (Line {item['line_number']})")

    def save_markdown(self, content: str, output_path: Path) -> None:
        """
        Save markdown content to a file.

        Args:
            content (str): Markdown content to save
            output_path (Path): Path where to save the file
        """
        try:
            output_path.write_text(content, encoding='utf-8')
            print(f"Markdown saved to: {output_path}")
        except Exception as e:
            raise Exception(f"Error saving markdown: {str(e)}")

    def process_pdf_complete(self, pdf_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Complete processing: PDF to markdown, extract TOC, and save files.

        Args:
            pdf_path (str): Path to input PDF file
            output_dir (str): Directory to save output files (optional)

        Returns:
            Dict: Processing results including paths and TOC data
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Set output directory
        if output_dir is None:
            output_dir = pdf_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Convert PDF to markdown
        print("Converting PDF to markdown...")
        markdown_content = self.pdf_to_markdown(pdf_path)

        # Extract table of contents
        print("Extracting table of contents...")
        toc = self.extract_table_of_contents(markdown_content)

        # Generate TOC markdown
        toc_markdown = self.generate_toc_markdown(toc)

        # Save files
        markdown_file = output_dir / f"{pdf_path.stem}_content.md"
        toc_file = output_dir / f"{pdf_path.stem}_toc.md"

        self.save_markdown(markdown_content, markdown_file)
        self.save_markdown(toc_markdown, toc_file)

        # Print structure
        self.print_toc_structure(toc)

        return {
            'markdown_content': markdown_content,
            'toc': toc,
            'toc_markdown': toc_markdown,
            'markdown_file': str(markdown_file),
            'toc_file': str(toc_file),
            'total_headings': len(toc)
        }


# Example usage
def main():
    """
    Example usage of the PDFToMarkdownProcessor class.
    """
    # Initialize processor with your Mistral API key
    api_key = "j3QqwjYyXHa692fQKr3hawypMuYCdRmE"  # Replace with your actual API key
    processor = PDFToMarkdownProcessor(api_key)

    try:
        # Process a PDF file
        pdf_file = r"/home/meghana/Downloads/BL9000-Owners-Manual.pdf"  # Replace with your PDF file path
        results = processor.process_pdf_complete(pdf_file)

        print(f"\nProcessing completed successfully!")
        print(f"Found {results['total_headings']} headings")
        print(f"Markdown content saved to: {results['markdown_file']}")
        print(f"Table of contents saved to: {results['toc_file']}")

        # You can also access the data programmatically
        print("\nFirst few headings:")
        for heading in results['toc'][:5]:  # Show first 5 headings
            print(f"  Level {heading['level']}: {heading['text']}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()