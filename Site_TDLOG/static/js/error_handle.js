document.addEventListener('DOMContentLoaded', () => { 
    const triggers = document.querySelectorAll('[aria-haspopup="dialog"]');
  
    triggers.forEach((trigger) => {
      const dialog = document.getElementById(trigger.getAttribute('aria-controls'));
    });
  });

  document.addEventListener('DOMContentLoaded', () => { 
    const triggers = document.querySelectorAll('[aria-haspopup="dialog"]');
  
    const open = function (dialog) {
      dialog.setAttribute('aria-hidden', false);
    };
  
    triggers.forEach((trigger) => {
      const dialog = document.getElementById(trigger.getAttribute('aria-controls'));
  
      // open dialog
      trigger.addEventListener('click', (event) => {
        event.preventDefault();
  
        open(dialog);
      });
    });
  });

  //Deactivate the main 
  ocument.addEventListener('DOMContentLoaded', () => { 
    const triggers = document.querySelectorAll('[aria-haspopup="dialog"]');
    const doc = document.querySelector('.js-document');
  
    const open = function (dialog) {
      dialog.setAttribute('aria-hidden', false);
      doc.setAttribute('aria-hidden', true);
    };
  
    triggers.forEach((trigger) => {
      const dialog = document.getElementById(trigger.getAttribute('aria-controls'));
  
      // open dialog
      trigger.addEventListener('click', (event) => {
        event.preventDefault();
  
        open(dialog);
      });
    });
  });

  //Close
  const close = function (dialog) {
    dialog.setAttribute('aria-hidden', true);
    doc.setAttribute('aria-hidden', false);
  };

  const dismissTriggers = dialog.querySelectorAll('[data-dismiss]');

// close dialog
dismissTriggers.forEach((dismissTrigger) => {
  const dismissDialog = document.getElementById(dismissTrigger.dataset.dismiss);

  dismissTrigger.addEventListener('click', (event) => {
    event.preventDefault();

    close(dismissDialog);
    });
}); 

//The user can also close by clicking behind
window.addEventListener('click', (event) => {
    if (event.target === dialog) {
      close(dialog);
    }
  });

  //Focus keyboard
  const focusableElementsArray = [
    '[href]',
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
  ];
  
  const open = function (dialog) {
    const focusableElements = dialog.querySelectorAll(focusableElementsArray);
  
    dialog.setAttribute('aria-hidden', false);
    doc.setAttribute('aria-hidden', true);
  };

  const open = function (dialog) {
    const focusableElements = dialog.querySelectorAll(focusableElementsArray);
    const firstFocusableElement = focusableElements[0];
  
    dialog.setAttribute('aria-hidden', false);
    doc.setAttribute('aria-hidden', true);
  
    // return if no focusable element
    if (!firstFocusableElement) {
      return;
    }
  
    window.setTimeout(() => {
      firstFocusableElement.focus();
    }, 100);
  }

  //Warn
  close(dialog, trigger);
  const close = function (dialog, trigger) {
    dialog.setAttribute('aria-hidden', true);
    doc.setAttribute('aria-hidden', false);
  
    // restoring focus
    trigger.focus();
  };