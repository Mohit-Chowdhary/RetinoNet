// Mobile Menu Toggle
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const mainNav = document.querySelector('.main-nav');

if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', () => {
        mobileMenuBtn.classList.toggle('active');
        mainNav.classList.toggle('mobile-open');
    });
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add mobile menu styles dynamically
if (window.innerWidth <= 768) {
    const style = document.createElement('style');
    style.textContent = `
        .main-nav {
            position: fixed;
            top: 60px;
            left: 0;
            right: 0;
            background: var(--bg-primary);
            flex-direction: column;
            padding: var(--spacing-lg);
            border-bottom: 1px solid var(--border-color);
            transform: translateY(-100%);
            opacity: 0;
            transition: all 0.3s ease;
            pointer-events: none;
        }
        
        .main-nav.mobile-open {
            transform: translateY(0);
            opacity: 1;
            pointer-events: all;
            display: flex;
        }
        
        .mobile-menu-btn.active span:nth-child(1) {
            transform: rotate(45deg) translate(5px, 5px);
        }
        
        .mobile-menu-btn.active span:nth-child(2) {
            opacity: 0;
        }
        
        .mobile-menu-btn.active span:nth-child(3) {
            transform: rotate(-45deg) translate(7px, -6px);
        }
    `;
    document.head.appendChild(style);
}

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe feature items and stats
document.querySelectorAll('.feature-item, .stat-item, .feature-detail-item').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// Enhanced Light Wave Effect Control
document.addEventListener('DOMContentLoaded', () => {
    const heroSection = document.querySelector('.hero');
    const lightBeam = document.querySelector('.light-beam--entry');
    const lightScatter = document.querySelector('.light-scatter--exit');
    const bgEffect = document.querySelector('.scattered-bg-effect');
    
    // Start animation when hero is in viewport
    const heroObserverOptions = {
        threshold: 0.3,
        rootMargin: '0px'
    };
    
    const heroObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Trigger animations
                if (lightBeam) lightBeam.style.animationPlayState = 'running';
                if (lightScatter) lightScatter.style.animationPlayState = 'running';
                if (bgEffect) bgEffect.style.animationPlayState = 'running';
            }
        });
    }, heroObserverOptions);
    
    if (heroSection) {
        heroObserver.observe(heroSection);
    }
    
    // Optional: Trigger enhanced effect on "Get Started" button hover
    const getStartedBtn = document.querySelector('.hero-actions .btn--primary');
    if (getStartedBtn) {
        getStartedBtn.addEventListener('mouseenter', () => {
            if (lightBeam) {
                lightBeam.style.animation = 'none';
                setTimeout(() => {
                    lightBeam.style.animation = 'beamEntry 2s ease-in-out forwards';
                }, 10);
            }
            if (lightScatter) {
                lightScatter.style.animation = 'none';
                setTimeout(() => {
                    lightScatter.style.animation = 'scatterExit 2s ease-in-out forwards';
                }, 10);
            }
        });
    }
});

// Add parallax effect to hero content
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const heroContent = document.querySelector('.hero-content');
    const heroImage = document.querySelector('.hero-image-wrapper');
    
    if (heroContent && window.innerWidth > 768) {
        heroContent.style.transform = `translateY(${scrolled * 0.3}px)`;
    }
    
    if (heroImage && window.innerWidth > 768) {
        heroImage.style.transform = `translateY(${scrolled * 0.2}px)`;
    }
});
// Add at the end of home.js

// Header scroll effect
const header = document.querySelector('.global-header');

window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        header.classList.add('scrolled');
    } else {
        header.classList.remove('scrolled');
    }
});
