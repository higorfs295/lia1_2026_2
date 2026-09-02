/* Comportamentos da interface — os mesmos do js/script.js do Dtox
   (menu fixo, rolagem suave, AOS), escritos sem jQuery para que a
   aplicação também rode no modo enxuto, sem os plugins do template. */
(function () {
  'use strict';

  var nav = document.querySelector('.navigation');
  function fixarMenu() {
    if (!nav) return;
    if (window.scrollY > 100) nav.classList.add('nav-bg');
    else nav.classList.remove('nav-bg');
  }
  window.addEventListener('scroll', fixarMenu);
  fixarMenu();

  // rolagem suave para as âncoras internas
  document.querySelectorAll('a[href*="#"]:not([href="#"])').forEach(function (a) {
    a.addEventListener('click', function (ev) {
      var id = a.getAttribute('href').split('#')[1];
      if (!id) return;
      var alvo = document.getElementById(id);
      if (!alvo) return;
      ev.preventDefault();
      window.scrollTo({ top: alvo.offsetTop - 90, behavior: 'smooth' });
    });
  });

  // menu no celular, sem o JS do Bootstrap
  var botao = document.querySelector('.navbar-toggler');
  if (botao) {
    botao.addEventListener('click', function () {
      var alvo = document.querySelector(botao.getAttribute('data-target') || '#navbar');
      if (alvo) alvo.classList.toggle('show');
    });
  }

  if (window.AOS) { AOS.init({ duration: 700, once: true }); }
})();
