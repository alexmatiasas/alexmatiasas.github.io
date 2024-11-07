---
layout: single
title: "Contact"
permalink: /contact/
author_profile: true
---
<!-- {: .text-justify} -->
Feel free to reach out through [LinkedIn](https://linkedin.com/in/alexmatiasastorga) or send me an email at [alejandromatiasastorga@gmail.com](mailto: alejandromatiasastorga@gmail.com). I'm always open to discussing new projects, creative ideas, or opportunities to be part of your vision.

[Download my CV](../CV/Manuel Alejandro Matías Astorga CV ENG.pdf){: .btn .btn--inverse}

### Contact form

<form action="https://formspree.io/f/{{ site.formspree_form_id }}" method="POST">
  <label for="name">Your name:</label>
  <input type="text" name="name" id="name" required>

  <label for="email">Your email:</label>
  <input type="email" name="email" id="email" required>

  <label for="message">Message:</label>
  <textarea name="message" id="message" required></textarea>

  <button type="submit">Send</button>
</form>