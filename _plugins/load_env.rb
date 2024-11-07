require 'dotenv'
Dotenv.load

Jekyll::Hooks.register :site, :pre_render do |site|
    site.config['formspree_form_id'] = ENV['FORMSPREE_FORM_ID']
  end