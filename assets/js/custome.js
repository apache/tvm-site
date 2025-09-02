$(document).ready(function () {
  $('#menuBtn').click(function () {
    $('#headMenu').addClass('open');
    $('body').addClass('scroll-hide');
  });
  $('#closeHeadMenu').click(function () {
    $('#headMenu').removeClass('open');
    $('body').removeClass('scroll-hide');
  });
  $('#openMailingModal').click(function () {
    $('#mailingModal').addClass('open');
  });
  $('#closeModal').click(function () {
    $('#mailingModal').removeClass('open');
  });
});
$(window).scroll(function () {
  var sticky = $('.header'),
    scroll = $(window).scrollTop();

  if (scroll >= 40) {
    sticky.addClass('fixed');
  }
  else {
    sticky.removeClass('fixed');
  }
});
