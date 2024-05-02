import Reveal from '/prod/reveal.js/dist/reveal.esm.js';
import Markdown from '/prod/reveal.js/plugin/markdown/markdown.esm.js';
import Highlight from '/prod/reveal.js/plugin/highlight/highlight.esm.js';
import { injectImages, injectScripts } from '/prod/gum.js/dist/js/gum.js';

// katex config
let delim = [
    {left: "$", right: "$", display: false},
    {left: "\\[", right: "\\]", display: true}
];

// autorender function
function autoRender(elem) {
    renderMathInElement(elem, {
        'delimiters': delim,
        'macros': {
            '\\eps': '\\varepsilon',
            '\\p': '\\prime',
            '\\fr': '\\frac{#1}{#2}',
            '\\pfr': '\\left(\\frac{#1}{#2}\\right)',
            '\\bfr': '\\left[\\frac{#1}{#2}\\right]',
            '\\cfr': '\\left\\{\\frac{#1}{#2}\\right\\}',
            '\\der': '\\frac{d#1}{d#2}',
            '\\pder': '\\frac{\\partial #1}{\\partial #2}',
            '\\eder': '\\frac{\\varepsilon #1}{\\varepsilon #2}',
            '\\E': '\\mathbb{E}\\br{#1}',
            '\\gr': '\\frac{\\dot{#1}}{#1}',
            '\\Ra': '\\Rightarrow',
            '\\ra': '\\rightarrow',
            '\\Ras': '\\ \\Rightarrow\\ ',
            '\\ras': '\\ \\rightarrow\\ ',
            '\\Raq': '\\quad\\Rightarrow\\quad',
            '\\raq': '\\quad\\rightarrow\\quad',
            '\\pr': '\\left(#1\\right)',
            '\\br': '\\left[#1\\right]',
            '\\cb': '\\left\\{#1\\right\\}',
            '\\qand': '\\quad\\text{and}\\quad',
            '\\where': '\\quad\\text{where}\\quad',
            '\\st': '\\text{s.t.}'
        }
    });
}

function makeLink(href) {
    let link = document.createElement('link');
    link.rel = 'stylesheet';
    link.type = 'text/css';
    link.href = href;
    return link;
}

function autoHooks(target) {
    autoRender(target);
    injectImages(target);
    injectScripts(target);
}

function initSlides() {
    // parse url args
    let urlParams = new URLSearchParams(window.location.search);
    let print_pdf = urlParams.has('print-pdf');

    // add in print styling
    if (print_pdf) {
        let head = document.getElementsByTagName('head')[0];
        let link1 = makeLink('/reveal.js/css/print/pdf.css');
        let link2 = makeLink('css/pdf.css');
        head.appendChild(link1);
        head.appendChild(link2);
        Reveal.addEventListener('ready', function() {
            autoHooks(document.body);
        });
    }

    // process equations in slides when they turn visible
    Reveal.addEventListener('slidechanged', function(event) {
        autoHooks(event.currentSlide);
    });

    Reveal.initialize({
        controls: false,
        progress: true,
        history: true,
        center: false,
        slideNumber: true,
        minScale: 0.1,
        margin: 0.1,
        transition: 'none',
        plugins: [ Markdown, Highlight ],
    });
}

export { initSlides }
