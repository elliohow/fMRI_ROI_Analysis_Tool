import dominate
import os
from dominate.tags import *
from glob import glob
from .utils import Utils


def create_index():
    doc = doc_setup("index.html")

    with doc.body:
        navbar(index=True)

        with div(cls='container'):
            with div(id='content'):
                for i in range(5):
                    br()

                h2('Figures')

                br()

                with table():
                    with thead():
                        with tr():
                            pass

                    with tbody():
                        # Find all figure directories that contain png images
                        fig_types = [f for f in glob(f"Figures/*") if glob(f"{f}/**/*.png", recursive=True)]

                        for fig in fig_types:
                            l = tr()
                            l.add(td(a(h4(str_format(os.path.split(fig)[1])), href=f'fRAT_report/{os.path.split(fig)[1]}.html')))
                            image = glob(f"{fig}/**/*.png", recursive=True)[0]
                            l.add(td(a(img(src=f"{image}", width=400), href=f'fRAT_report/{os.path.split(fig)[1]}.html')))

    save_page(doc, 'index.html')


def create_figure_pages(figure_type):
    folders = [f for f in glob(f"{figure_type}/*/", recursive=True)]

    if len(folders):
        figures = []
        for folder in folders:
            figures.append([f for f in glob(f"{folder}*.png", recursive=True)])

    else:
        folders = [figure_type]
        figures = [f for f in glob(f"{folders[0]}/**/*.png", recursive=True)]

    for counter, folder in enumerate(folders):
        doc = doc_setup()
        axis_type = os.path.split(folders[counter][:-1])[1]

        with doc.body:
            navbar()

            with div(cls='container'):
                with div(id='content'):
                    for i in range(5):
                        br()

                    if len(folders) > 1:
                        folder = os.path.split(os.path.split(folder[:-1])[0])[1]
                        h2(f"{folder} ({str_format(axis_type).lower()})")
                    else:
                        folder = os.path.split(folder)[1]
                        h2(f"{str_format(folder)}")

                    br()

                    with table():
                        with tbody():
                            if len(folders) > 1:
                                plot_figs(folder, figures[counter])
                            else:
                                plot_figs(folder, figures)

        if len(folders) > 1:
            save_page(doc, f'{os.path.split(figure_type)[1]}_{axis_type}.html')
        else:
            save_page(doc, f'{os.path.split(figure_type)[1]}.html')

    if len(folders) > 1:
        doc = doc_setup()

        with doc.body:
            navbar()

            with div(cls='container'):
                with div(id='content'):
                    for i in range(5):
                        br()

                    h2(os.path.split(figure_type)[1])

                    br()

                    with table():
                        with tbody():
                            l = tr()

                            for folder in folders:
                                l.add(a(h4(str_format(os.path.split(folder[:-1])[1])),
                                        href=f'{os.path.split(figure_type)[1]}_{os.path.split(folder[:-1])[1]}.html'))

        save_page(doc, f'{os.path.split(figure_type)[1]}.html')


def str_format(string):
    replace_chars = (("axis", "-axis"),
                     ("roi", "ROI"),
                     ("stat", "Stat"),
                     ("same_ylim", ""),
                     ("same_xlim", ""),
                     ('_', ' '))

    for char in replace_chars:
        string = string.replace(char[0], char[1])

    return string


def plot_figs(figure_type, figures):
    # Determine how much to trim off the end of the file name (removing .png etc.)
    if figure_type in ('Brain_grids', 'Brain_images'):
        length = 1
    else:
        length = len(figure_type)

    for fig in figures:
        fig_name = os.path.split(str_format(fig))[1][:-(length+3)]  # Format name for better presentation

        l = tr()
        l.add(td(a(h4(str_format(fig_name)), href=f"../{fig}")))  # Add plot title
        l.add(td(a(img(src=f"../{fig}", width=700), href=f"../{fig}")))  # Show plot


def doc_setup(page=None):
    if page == 'index.html':
        folder = "fRAT_report/"
    else:
        folder = ""

    doc = dominate.document(title='fRAT report')

    with doc.head:
        link(rel='stylesheet', href=f'{folder}bootstrap.css')
        script(type='text/javascript', src=f'{folder}script.js')

    return doc


def save_page(doc, page):
    if page == 'index.html':
        folder = ""
    else:
        folder = "fRAT_report/"

    with open(f"{folder}{page}", 'w') as file:
        file.write(doc.render())


def navbar(index=False):
    with nav(cls="navbar navbar-expand-lg fixed-top navbar-dark bg-primary"):
        with div(cls='container'):
            span('fRAT report', style="font-size:30px", cls="navbar-brand")
            span(id="output", style="font-size:15px", cls="navbar-brand")

            if index:
                with div(cls='collapse navbar-collapse', id='navbarResponsive'):
                    with ul(cls='navbar-nav'):
                        li()  # Prevent flex from forcing folder name to the right hand side on index page

                script("""
                       var loc = window.location.pathname;
                       document.getElementById('output').innerHTML = '/' + loc.split('/').slice(-2, -1);
                       """)
            else:
                with div(cls='collapse navbar-collapse', id='navbarResponsive'):
                    with ul(cls='navbar-nav'):
                        li(a('Back to Figures', cls="nav-link", href=f'../index.html', align='center'), cls='nav-item')

                script("""
                       var loc = window.location.pathname;
                       document.getElementById('output').innerHTML = '/' + loc.split('/').slice(-3, -2);
                       """)


def main(orig_path):
    Utils.check_and_make_dir("fRAT_report")
    orig_path += "/utils"
    Utils.move_file('bootstrap.css', orig_path, 'fRAT_report/', copy=True, rename_copy=False)
    Utils.move_file('script.js', orig_path, 'fRAT_report/', copy=True, rename_copy=False)

    create_index()
    fig_types = [f for f in glob(f"Figures/*") if f != 'Figures/figure_log.toml']

    for fig in fig_types:
        create_figure_pages(fig)
