# Manuscript

`main.pdf` contains 18 pages of body text. Data Availability starts at the top of
page 19, followed by the references. The complete PDF has 21 pages.

The source uses the official ACM `acmsmall,screen,review,anonymous` format;
the bundled class and bibliography style retain their original license.
Compile from this directory with an existing LaTeX distribution, for example
`latexmk -xelatex main.tex`, or `tectonic main.tex`. All used tables and figures
are included. No separate supplementary PDF is required to follow the paper.

Machine-readable evidence and the statistical reproduction entry point are in
`../evidence/`. Model weights and their checksums are in `../frozen/`.
